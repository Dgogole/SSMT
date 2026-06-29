import torch
import torch.nn as nn
import csv
import json
import math
from utils import builder
from utils.AverageMeter import AverageMeter
import numpy as np
from einops import rearrange
from utils.rotation import rotation_6d_to_matrix
import os
from tqdm import tqdm


# =============================================================================
# Utility Functions
# =============================================================================

def _sanitize_sample_id(sample_id):
    return str(sample_id).replace(os.sep, '_').replace(' ', '_')


def _write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Metric Calculation
# =============================================================================


def _rotation_angle_degrees(rotation_matrix):
    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / math.pi)


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def predict_ssmt_points(before_points, centroid, outputs):
    predicted_centroid = outputs[:, :, :3]
    bsz, num_teeth, npoint, _ = before_points.shape
    rot6d = rearrange(outputs[:, :, 3:], 'b n c -> (b n) c').float()
    rot_matrix = rotation_6d_to_matrix(rot6d)
    predicted_points = rearrange(before_points - centroid.unsqueeze(2), 'b n p c -> (b n) p c')
    predicted_points = torch.bmm(predicted_points, rot_matrix)
    predicted_points = predicted_points + rearrange(predicted_centroid, 'b n c -> (b n) c').unsqueeze(1)
    predicted_points_bnpc = rearrange(
        predicted_points,
        '(b n) p c -> b n p c',
        b=bsz,
        n=num_teeth,
    )
    return predicted_points, predicted_points_bnpc


def compute_ssmt_losses(outputs, before_points, after_points, centroid, gt_params, masks, config):
    criterion = nn.MSELoss(reduction='none')
    loss_config = config.get('loss', {}) if hasattr(config, 'get') else getattr(config, 'loss', {})
    point_weight = float(_cfg_get(loss_config, 'point_weight', 0.001))
    rotation_weight = float(_cfg_get(loss_config, 'rotation_weight', _cfg_get(loss_config, 'param_weight', 0.03)))
    centroid_weight = float(_cfg_get(loss_config, 'centroid_weight', 0.001))
    valid_masks = masks.float()
    valid_count = valid_masks.sum(dim=1).clamp(min=1.0)

    predicted_points, _ = predict_ssmt_points(before_points, centroid, outputs)
    after_points_flat = rearrange(after_points, 'b n p c -> (b n) p c')
    point_error_sq = criterion(predicted_points, after_points_flat).sum(dim=-1)
    bsz, num_teeth, _ = after_points.shape[:3]
    tooth_point_mse = point_error_sq.mean(dim=1).reshape(bsz, num_teeth)
    pointcloud_loss = point_weight * ((tooth_point_mse * valid_masks).sum(dim=1) / valid_count).mean()

    tooth_rotation_loss = criterion(outputs[:, :, 3:], gt_params[:, :, 3:]).sum(dim=-1)
    rotation_loss = rotation_weight * ((tooth_rotation_loss * valid_masks).sum(dim=1) / valid_count).mean()
    tooth_centroid_loss = criterion(outputs[:, :, :3], gt_params[:, :, :3]).sum(dim=-1)
    centroid_loss = centroid_weight * ((tooth_centroid_loss * valid_masks).sum(dim=1) / valid_count).mean()
    loss = pointcloud_loss + rotation_loss + centroid_loss
    return loss, [
        loss.item(),
        pointcloud_loss.item(),
        rotation_loss.item(),
        centroid_loss.item(),
    ]


def compute_pointcloud_metrics(after_points, pred_points, outputs, gt_params, masks, scale=40.0):
    valid_masks = masks.bool()
    bsz, num_teeth = valid_masks.shape

    point_dist = torch.linalg.norm(pred_points - after_points, dim=-1)
    tooth_mean_point_error = point_dist.mean(dim=-1) * scale

    pred_rot = rotation_6d_to_matrix(outputs[:, :, 3:].reshape(bsz * num_teeth, 6)).reshape(bsz, num_teeth, 3, 3)
    gt_rot = rotation_6d_to_matrix(gt_params[:, :, 3:].reshape(bsz * num_teeth, 6)).reshape(bsz, num_teeth, 3, 3)
    relative_rot = pred_rot.matmul(gt_rot.transpose(-1, -2))
    tooth_me_rot = _rotation_angle_degrees(relative_rot)
    tooth_me_trans = torch.linalg.norm(outputs[:, :, :3] - gt_params[:, :, :3], dim=-1) * scale

    sample_metrics = []
    for batch_idx in range(bsz):
        valid = valid_masks[batch_idx]
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            sample_metrics.append({'ADD_TRE': 0.0, 'MEtrans': 0.0, 'MErot': 0.0})
            continue

        sample_metrics.append({
            'ADD_TRE': float(tooth_mean_point_error[batch_idx, valid].mean().item()),
            'MEtrans': float(tooth_me_trans[batch_idx, valid].mean().item()),
            'MErot': float(tooth_me_rot[batch_idx, valid].mean().item()),
        })

    return sample_metrics


def export_pointcloud_batch(
    sample_ids,
    outputs,
    after_points,
    pred_points,
    masks,
    sample_metrics,
    sample_rows,
    prediction_root,
):
    bsz, num_teeth = outputs.shape[:2]
    pred_params_np = outputs.detach().cpu().numpy()
    after_np = after_points.detach().cpu().numpy()
    pred_np = pred_points.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy().astype(bool)

    for batch_idx, sample_id in enumerate(sample_ids):
        sample_id = str(sample_id)
        case_dir = os.path.join(prediction_root, _sanitize_sample_id(sample_id))
        os.makedirs(case_dir, exist_ok=True)
        metric_valid = masks_np[batch_idx]
        valid_teeth = int(metric_valid.sum())

        if metric_valid.any():
            after_valid_points = after_np[batch_idx, metric_valid].reshape(-1, 3)
            pred_valid = pred_np[batch_idx, metric_valid].reshape(-1, 3)
            per_point_error = np.sqrt(((pred_valid - after_valid_points) ** 2).sum(axis=-1))
            mean_point_error = float(per_point_error.mean())
        else:
            mean_point_error = 0.0

        np.save(os.path.join(case_dir, 'pred_params_9d.npy'), pred_params_np[batch_idx])

        sample_rows.append({
            'sample_id': sample_id,
            'before_teeth': valid_teeth,
            'after_teeth': valid_teeth,
            'supervised_teeth': valid_teeth,
            'mean_point_error': mean_point_error,
            'ADD_TRE': sample_metrics[batch_idx]['ADD_TRE'],
            'MEtrans': sample_metrics[batch_idx]['MEtrans'],
            'MErot': sample_metrics[batch_idx]['MErot'],
            'case_dir': case_dir,
        })


# =============================================================================
# Test Entry
# =============================================================================

def test_ssmt(args, config, logger):
    # --- Build Dataset & Model ---
    config.model.args = args
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model_from_ckpt(base_model, args.ckpts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        base_model = nn.DataParallel(base_model).to(device)
    else:
        base_model = base_model.to(device)
    base_model.eval()

    # --- Evaluate ---
    export_root = os.path.join(args.experiment_path, 'eval')
    os.makedirs(export_root, exist_ok=True)
    prediction_root = os.path.join(args.experiment_path, 'predictions')
    os.makedirs(prediction_root, exist_ok=True)
    metric_meters = AverageMeter(['ADD_TRE', 'MEtrans', 'MErot'])
    loss_sum = 0.0
    sample_count = 0
    sample_rows = []

    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing", unit="batch")
        for _idx, (
            index, feats, center, cordinates, faces, Fs, before_points, after_points,
            centroid, _after_centroid, gt_params, masks, text_feature,
        ) in pbar:
            faces = faces.to(device=device, dtype=torch.float32)
            feats = feats.to(device=device, dtype=torch.float32)
            center = center.to(device=device, dtype=torch.float32)
            gt_params = gt_params.to(device=device, dtype=torch.float32)
            Fs = Fs.to(device=device)
            cordinates = cordinates.to(device=device)
            centroid = centroid.to(device=device, dtype=torch.float32)
            before_points = before_points.to(device=device, dtype=torch.float32)
            after_points = after_points.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device).bool()
            text_feature = text_feature.to(device=device, dtype=torch.float32)

            model_out = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, text_feature, gt_params)
            outputs = model_out[0] if isinstance(model_out, (tuple, list)) else model_out
            outputs = outputs.to(device=device, dtype=torch.float32)

            loss, stats = compute_ssmt_losses(outputs, before_points, after_points, centroid, gt_params, masks, config)
            _, predicted_points_bnpc = predict_ssmt_points(before_points, centroid, outputs)
            sample_metrics = compute_pointcloud_metrics(
                after_points,
                predicted_points_bnpc,
                outputs,
                gt_params,
                masks,
            )

            batch_size = len(sample_metrics)
            loss_sum += stats[0] * batch_size
            sample_count += batch_size
            for metrics in sample_metrics:
                metric_meters.update([metrics['ADD_TRE'], metrics['MEtrans'], metrics['MErot']])

            avg_metrics = metric_meters.avg()
            pbar.set_postfix({
                'ADD_TRE': f'{avg_metrics[0]:.4f}',
                'MEtrans': f'{avg_metrics[1]:.4f}',
                'MErot': f'{avg_metrics[2]:.4f}',
            })

            export_pointcloud_batch(
                index,
                outputs,
                after_points,
                predicted_points_bnpc,
                masks,
                sample_metrics,
                sample_rows,
                prediction_root,
            )

    # --- Export Results ---
    mean_loss = loss_sum / max(sample_count, 1)
    avg_metrics = metric_meters.avg()
    _write_csv(
        os.path.join(export_root, 'sample_metrics.csv'),
        sample_rows,
        fieldnames=[
            'sample_id', 'before_teeth', 'after_teeth', 'supervised_teeth',
            'mean_point_error', 'ADD_TRE', 'MEtrans', 'MErot', 'case_dir',
        ],
    )
    summary = {
        'mean_loss': float(mean_loss),
        'mean_ADD_TRE': float(avg_metrics[0]),
        'mean_MEtrans': float(avg_metrics[1]),
        'mean_MErot': float(avg_metrics[2]),
        'num_samples': len(sample_rows),
        'export_root': export_root,
        'prediction_root': prediction_root,
    }
    _write_json(os.path.join(export_root, 'summary.json'), summary)

    logger.info(
        '[Test] mean_loss=%.4f ADD_TRE=%.4f MEtrans=%.4f MErot=%.4f'
        % (mean_loss, avg_metrics[0], avg_metrics[1], avg_metrics[2])
    )
    logger.info(f"[Test] results saved to {export_root}")
    print("Final Test Results:")
    print(f"  mean_loss={mean_loss:.4f}")
    print(f"  Mean ADD/TRE:  {avg_metrics[0]:.4f}")
    print(f"  Mean MEtrans:  {avg_metrics[1]:.4f}")
    print(f"  Mean MErot:    {avg_metrics[2]:.4f}")
    print(f"Results: {export_root}")
