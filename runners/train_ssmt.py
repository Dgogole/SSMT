import torch
import torch.nn as nn
from utils import builder
import time
from utils.AverageMeter import AverageMeter
from einops import rearrange
from utils.rotation import rotation_6d_to_matrix


# =============================================================================
# Loss Helpers
# =============================================================================

def get_matrix(centroid):
    return torch.cdist(centroid, centroid)


def get_matrix_mask(masks):
    f_masks = masks.float()
    matrix_mask = torch.bmm(f_masks.unsqueeze(-1), f_masks.unsqueeze(-2))
    return matrix_mask


def compute_add_tre(point_error_sq, masks, batch_size, num_teeth, scale=40.0):
    point_error = point_error_sq.sqrt().reshape(batch_size, num_teeth, -1)
    tooth_mean_point_error = point_error.mean(dim=-1) * scale
    valid_masks = masks.float()
    valid_count = valid_masks.sum(dim=1).clamp(min=1.0)
    return ((tooth_mean_point_error * valid_masks).sum(dim=1) / valid_count).mean()


def pairwise_min_distances_adjacent_vectorized(P_batch, masks, eps=1e-6):
    """
    Collision loss between adjacent teeth.

    Args:
        P_batch: [B, N, P, 3] predicted point clouds
        masks: [B, N] tooth-valid mask
    Returns:
        loss: scalar tensor
    """
    masks = masks > 0.5
    pairs = []

    for b in range(P_batch.shape[0]):
        valid = torch.nonzero(masks[b], as_tuple=False).squeeze(-1)
        valid = valid[~((valid == 15) | (valid == 31))]
        if valid.numel() < 2:
            continue
        i = valid[:-1]
        j = valid[1:]
        b_idx = torch.full_like(i, b)
        pairs.append(torch.stack([b_idx, i, j], dim=1))

    if not pairs:
        zero = P_batch.sum() * 0.0
        return zero

    all_pairs = torch.cat(pairs, dim=0)
    b_idx, i_idx, j_idx = all_pairs[:, 0], all_pairs[:, 1], all_pairs[:, 2]

    pc1 = P_batch[b_idx, i_idx]
    pc2 = P_batch[b_idx, j_idx]
    dists = torch.cdist(pc1, pc2)
    min_dists = torch.amin(dists, dim=(1, 2)) + eps

    c1 = torch.mean(pc1, dim=1)
    c2 = torch.mean(pc2, dim=1)
    center_dists = torch.norm(c1 - c2, dim=-1)

    pair_losses = (min_dists / (min_dists + center_dists + eps)).pow(2)
    loss = pair_losses.sum()
    return loss


# =============================================================================
# Training
# =============================================================================

def train_ssmt(args, config, train_writer, val_writer, logger, accelerator):
    # --- Build Dataset & Model ---
    config.model.args = args
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)

    loss_cfg = config.get('loss', None)
    collision_lambda = float(getattr(loss_cfg, 'collision_lambda', 1.0)) if loss_cfg is not None else 1.0
    collision_eps = float(getattr(loss_cfg, 'collision_eps', 1e-6)) if loss_cfg is not None else 1e-6

    # --- Resume / Load Checkpoint ---
    start_epoch = 0
    best_metrics = 99999

    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
    else:
        if args.ckpts != '':
            start_epoch, best_metrics = builder.resume_from_checkpoint(base_model, args, logger=logger)
        else:
            if accelerator.is_main_process:
                logger.info('Training from scratch')

    if accelerator.is_main_process:
        logger.info('Using Distributed Data Parallel via Accelerate')
        logger.info(f'[Collision] collision_lambda={collision_lambda:.6f} collision_eps={collision_eps:.2e}')

    # --- Optimizer & Scheduler ---
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    if args.ckpts != '':
        builder.resume_optimizer_from_checkpoint(optimizer, args, logger=logger)

    # --- Accelerator Prepare ---
    base_model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        base_model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # --- Training Loop ---
    for epoch in range(start_epoch, config.max_epoch + 1):
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        n_batches = len(train_dataloader)
        losses = AverageMeter([
            'loss', 'loss1', 'loss2', 'loss3', 'loss_collision', 'add_tre'
        ])
        criterion1 = nn.MSELoss(reduction='none')

        for idx, (index, feats, center, cordinates, faces, Fs, before_points, after_points, centroid, after_centroid, gt_params, masks, text_feature) in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            data_time.update(time.time() - batch_start_time)

            faces = faces.to(torch.float32)
            feats = feats.to(torch.float32)
            center = center.to(torch.float32)
            gt_params = gt_params.to(torch.float32)
            centroid = centroid.to(torch.float32)
            after_centroid = after_centroid.to(torch.float32)
            before_points = before_points.to(torch.float32)
            after_points = after_points.to(torch.float32)
            text_feature = text_feature.to(torch.float32)
            masks = masks.to(torch.float32)

            model_out = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, text_feature, gt_params)
            outputs = model_out[0].float() if isinstance(model_out, (tuple, list)) else model_out.float()
            predicted_centroid = outputs[:, :, :3]
            rot6d = rearrange(outputs[:, :, 3:], 'b n c -> (b n) c').float()
            rot_matrix = rotation_6d_to_matrix(rot6d)

            predicted_points = rearrange(before_points - centroid.unsqueeze(2), 'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points, rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_centroid, 'b n c->(b n) c').unsqueeze(1)
            predicted_points_bnpc = rearrange(
                predicted_points,
                '(b n) p c -> b n p c',
                b=before_points.shape[0],
                n=before_points.shape[1],
            )
            after_points = rearrange(after_points, 'b n p c -> (b n) p c')
            point_error_sq = criterion1(predicted_points, after_points).sum(dim=-1)

            # Metric (observation only, same formula as val)
            with torch.no_grad():
                add_tre = compute_add_tre(
                    point_error_sq,
                    masks,
                    batch_size=before_points.shape[0],
                    num_teeth=before_points.shape[1],
                )
            loss_collision = pairwise_min_distances_adjacent_vectorized(
                predicted_points_bnpc, masks, eps=collision_eps
            )
            loss_collision = collision_lambda * loss_collision

            loss1 = point_error_sq.sum(dim=1)
            loss1 = 0.001 * (loss1 * masks.flatten()).sum()
            loss2 = 0.03 * ((criterion1(outputs, gt_params).sum(dim=-1)) * masks).mean()
            loss3 = 0.001 * (criterion1(get_matrix(centroid), get_matrix(predicted_centroid)) * get_matrix_mask(masks)).sum()

            loss = loss1 + loss2 + loss3 + loss_collision

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.update([
                loss.item(), loss1.item(), loss2.item(), loss3.item(), loss_collision.item(),
                add_tre.item()
            ])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 5 == 0 and accelerator.is_main_process:
                logger.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) '
                            'Loss = %.4f [loss1=%.4f, loss2=%.4f, loss3=%.4f, loss_collision=%.4f] '
                            '| ADD_TRE = %.4f | lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            losses.val()[0], losses.val()[1], losses.val()[2], losses.val()[3], losses.val()[4],
                            losses.val()[5], optimizer.param_groups[0]['lr']))

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        # --- Epoch Logging ---
        if accelerator.is_main_process:
            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/Loss_PointCloud', losses.avg(1), epoch)
                train_writer.add_scalar('Loss/Epoch/Loss_Diffusion', losses.avg(2), epoch)
                train_writer.add_scalar('Loss/Epoch/Loss_Centroid', losses.avg(3), epoch)
                train_writer.add_scalar('Loss/Epoch/Loss_Collision', losses.avg(4), epoch)
                train_writer.add_scalar('Metric/Epoch/ADD_TRE', losses.avg(5), epoch)

            logger.info('[Training] EPOCH: %d EpochTime = %.3f (s) Loss = %.4f '
                        '[loss1=%.4f, loss2=%.4f, loss3=%.4f, loss_collision=%.4f] '
                        '| ADD_TRE = %.4f | lr = %.6f' %
                (epoch, epoch_end_time - epoch_start_time, losses.avg(0), losses.avg(1), losses.avg(2),
                 losses.avg(3), losses.avg(4), losses.avg(5), optimizer.param_groups[0]['lr']))

        # --- Validation & Checkpoint ---
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, accelerator, logger=logger)

            if accelerator.is_main_process:
                logger.info('[Validation] EPOCH: %d  Loss = %.4f' % (epoch, metrics))
                if metrics < best_metrics:
                    best_metrics = metrics
                    unwrapped_model = accelerator.unwrap_model(base_model)
                    builder.save_checkpoint(unwrapped_model, optimizer, epoch, best_metrics, 'ckpt-best', args, logger=logger)

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(base_model)
            builder.save_checkpoint(unwrapped_model, optimizer, epoch, best_metrics, 'ckpt-last', args, logger=logger)
            logger.info("--------------------------------------------------------------------------------------------")

    if accelerator.is_main_process:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()


# =============================================================================
# Validation
# =============================================================================

def validate(base_model, test_dataloader, epoch, val_writer, args, config, accelerator, logger=None):
    if accelerator.is_main_process:
        logger.info(f"[Validation] Start validating epoch {epoch}")

    loss_cfg = config.get('loss', None)
    collision_lambda = float(getattr(loss_cfg, 'collision_lambda', 1.0)) if loss_cfg is not None else 1.0
    collision_eps = float(getattr(loss_cfg, 'collision_eps', 1e-6)) if loss_cfg is not None else 1e-6
    base_model.eval()
    losses = AverageMeter([
        'loss', 'loss1', 'loss2', 'loss3', 'loss_collision', 'add_tre'
    ])
    with torch.no_grad():
        for idx, (index, feats, center, cordinates, faces, Fs, before_points, after_points, centroid, after_centroid, gt_params, masks, text_feature) in enumerate(test_dataloader):
            faces = faces.to(torch.float32)
            feats = feats.to(torch.float32)
            center = center.to(torch.float32)
            gt_params = gt_params.to(torch.float32)
            centroid = centroid.to(torch.float32)
            after_centroid = after_centroid.to(torch.float32)
            before_points = before_points.to(torch.float32)
            after_points = after_points.to(torch.float32)
            masks = masks.to(torch.float32)
            text_feature = text_feature.to(torch.float32)

            model_out = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, text_feature, gt_params)
            outputs = model_out[0].float() if isinstance(model_out, (tuple, list)) else model_out.float()

            predicted_translation = outputs[:, :, :3]
            rot6d = rearrange(outputs[:, :, 3:], 'b n c -> (b n) c')
            criterion = nn.MSELoss(reduction='none')
            rot_matrix = rotation_6d_to_matrix(rot6d)
            predicted_points = rearrange(before_points - centroid.unsqueeze(2), 'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points, rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_translation, 'b n c->(b n) c').unsqueeze(1)
            predicted_points_bnpc = rearrange(
                predicted_points,
                '(b n) p c -> b n p c',
                b=before_points.shape[0],
                n=before_points.shape[1],
            )
            after_points = rearrange(after_points, 'b n p c -> (b n) p c')

            point_error_sq = criterion(predicted_points, after_points).sum(dim=-1)
            add_tre = compute_add_tre(
                point_error_sq,
                masks,
                batch_size=before_points.shape[0],
                num_teeth=before_points.shape[1],
            )
            loss1 = point_error_sq.sum(dim=1)
            loss1 = 0.001 * (loss1 * masks.flatten()).sum()
            loss2 = 0.03 * ((criterion(outputs, gt_params).sum(dim=-1)) * masks).mean()
            loss3 = 0.001 * (
                criterion(get_matrix(centroid), get_matrix(predicted_translation)) * get_matrix_mask(masks)
            ).sum()
            loss_collision = pairwise_min_distances_adjacent_vectorized(
                predicted_points_bnpc, masks, eps=collision_eps
            )
            loss_collision = collision_lambda * loss_collision
            loss = loss1 + loss2 + loss3 + loss_collision

            losses.update([
                loss.item(), loss1.item(), loss2.item(), loss3.item(), loss_collision.item(), add_tre.item()
            ])

    # Aggregate across processes
    avg_losses = losses.avg()
    reduced = []
    for value in avg_losses:
        value_tensor = torch.tensor(value, device=accelerator.device)
        reduced.append(accelerator.reduce(value_tensor, reduction="mean").item())
    final_loss = reduced[5]

    if accelerator.is_main_process and val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Loss', reduced[0], epoch)
        val_writer.add_scalar('Loss/Epoch/Loss_PointCloud', reduced[1], epoch)
        val_writer.add_scalar('Loss/Epoch/Loss_Diffusion', reduced[2], epoch)
        val_writer.add_scalar('Loss/Epoch/Loss_Centroid', reduced[3], epoch)
        val_writer.add_scalar('Loss/Epoch/Loss_Collision', reduced[4], epoch)
        val_writer.add_scalar('Metric/Epoch/ADD_TRE', reduced[5], epoch)

    if accelerator.is_main_process and logger is not None:
        logger.info('[Validation] EPOCH: %d Loss = %.4f [loss1=%.4f, loss2=%.4f, loss3=%.4f, loss_collision=%.4f] '
                    '| ADD_TRE = %.4f' %
                    (epoch, reduced[0], reduced[1], reduced[2], reduced[3], reduced[4], reduced[5]))

    return final_loss
