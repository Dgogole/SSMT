import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import math
from einops import rearrange
from models.MeshMAE import Mesh_encoder
from models.pointnet import PointNetEncoder
from timm.models.vision_transformer import Block
from utils.builder import MODELS

# ================= Direct MLP Regressor =================
class DirectRegressorMLP(nn.Module):
    def __init__(self, in_features, num_teeth=32, out_dim=9):
        """
        Direct MLP regression head for 6DoF prediction

        Args:
            in_features: Input feature dimension (final_dim)
            num_teeth: Number of teeth (default: 32)
            out_dim: Output dimension, 9 for 6DoF (3 Translation + 6 Rotation)
        """
        super(DirectRegressorMLP, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, out_dim)  # Output [BS, 32, 9]
        )

    def forward(self, final_feat):
        """
        Args:
            final_feat: [BS, 32, in_features]
        Returns:
            [BS, 32, 9] - 6DoF parameters (3 translation + 6 rotation)
        """
        return self.regressor(final_feat)
# =========================================================


class CrossAttention(nn.Module):
    """
    Cross-Attention module for multimodal fusion (similar to MulT)
    Key/Value: main modality
    Query: secondary modality

    Supports different dimensions for main (K/V) and secondary (Q) via projection layers.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, key_dim=None):
        """
        Args:
            embed_dim: Query dimension (secondary modality)
            key_dim: Key/Value dimension (main modality), if None, use embed_dim
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim if key_dim is not None else embed_dim

        # Projection layers for key/value if dimensions differ
        if self.key_dim != self.embed_dim:
            self.key_proj = nn.Linear(self.key_dim, self.embed_dim)
            self.value_proj = nn.Linear(self.key_dim, self.embed_dim)
        else:
            self.key_proj = None
            self.value_proj = None

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, main_feat, secondary_feat):
        """
        Args:
            main_feat: [B, seq_len_main, key_dim] for Key/Value
            secondary_feat: [B, seq_len_secondary, embed_dim] for Query
        Returns:
            output: [B, seq_len_secondary, embed_dim]
        """
        query = secondary_feat
        key_value = main_feat

        # Project K/V if dimensions differ
        if self.key_proj is not None:
            key = self.key_proj(key_value)
            value = self.value_proj(key_value)
        else:
            key = key_value
            value = key_value

        # Cross-attention
        attn_out, _ = self.multihead_attn(query, key, value)
        query = self.norm(query + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


class SelfAttention(nn.Module):
    """
    Self-Attention module for intra-modal processing
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embed_dim]
        Returns:
            output: [B, seq_len, embed_dim]
        """
        # Self-attention
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SelfAttentionStack(nn.Module):
    """
    Stacked self-attention layers for deep intra-modal fusion.
    """
    def __init__(self, embed_dim, depth=1, num_heads=8, dropout=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.layers = nn.ModuleList([
            SelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FDISemanticPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, slot_to_fdi):
        super().__init__()
        self.jaw_embed = nn.Embedding(2, embed_dim)
        self.side_embed = nn.Embedding(2, embed_dim)
        self.midline_embed = nn.Embedding(8, embed_dim)
        self.semantic_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.register_buffer("slot_to_fdi", torch.as_tensor(slot_to_fdi, dtype=torch.long), persistent=False)

    def get_fdi_indices(self, tooth_indices):
        if tooth_indices.max().item() >= self.slot_to_fdi.numel() or tooth_indices.min().item() < 0:
            raise ValueError(
                f"tooth_indices must be in [0, {self.slot_to_fdi.numel() - 1}], "
                f"got min={tooth_indices.min().item()}, max={tooth_indices.max().item()}."
            )
        fdi_codes = self.slot_to_fdi[tooth_indices]
        quadrant = fdi_codes // 10
        tooth_num = fdi_codes % 10

        jaw_idx = ((quadrant == 3) | (quadrant == 4)).long()
        side_idx = ((quadrant == 1) | (quadrant == 4)).long()
        midline_idx = (tooth_num - 1).clamp(min=0, max=7).long()
        return jaw_idx, side_idx, midline_idx

    def forward(self, x, tooth_indices):
        jaw_idx, side_idx, midline_idx = self.get_fdi_indices(tooth_indices)
        jaw_pe = self.jaw_embed(jaw_idx)
        side_pe = self.side_embed(side_idx)
        midline_pe = self.midline_embed(midline_idx)
        combined_pe = self.semantic_proj(jaw_pe + side_pe + midline_pe)
        return x + combined_pe


class SpatialSinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embed_dim, num_frequencies=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        # 3 (x,y,z) * num_frequencies * 2 (sin, cos)
        spatial_in_dim = 3 * num_frequencies * 2
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_in_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, centroids, x):
        """
        centroids: [B, N, 3] physical 3D centroids
        x: [B, N, embed_dim] branch feature
        """
        if centroids.dim() != 3 or centroids.shape[-1] != 3:
            raise ValueError(f"centroids must have shape [B, N, 3], got {tuple(centroids.shape)}")
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, N, C], got {tuple(x.shape)}")
        if centroids.shape[0] != x.shape[0] or centroids.shape[1] != x.shape[1]:
            raise ValueError(
                f"centroids and x must match on [B, N], got centroids={tuple(centroids.shape)}, x={tuple(x.shape)}"
            )

        bsz, num_slots, _ = centroids.shape
        centroids_f = centroids.float()
        device = centroids_f.device

        # 2^0, 2^1, ..., 2^(L-1)
        freq_bands = (2 ** torch.arange(self.num_frequencies, device=device, dtype=centroids_f.dtype))

        # [B, N, 3, L]
        pts = centroids_f.unsqueeze(-1) * freq_bands
        pts_sin = torch.sin(pts * math.pi)
        pts_cos = torch.cos(pts * math.pi)

        # [B, N, 3, L] * 2 -> [B, N, 3*2*L]
        spatial_pe_raw = torch.cat([pts_sin, pts_cos], dim=-1).reshape(bsz, num_slots, -1)
        spatial_pe_projected = self.spatial_proj(spatial_pe_raw).to(dtype=x.dtype)
        return x + spatial_pe_projected


@MODELS.register_module()
class SSMT(nn.Module):
    def __init__(self,config):
        super(SSMT, self).__init__()
        self.debug = getattr(config, 'debug', False)
        self.debug_stats = getattr(config, 'debug_stats', True)
        self.debug_max_forwards = max(0, int(getattr(config, 'debug_max_forwards', 2)))
        self.debug_rank0_only = getattr(config, 'debug_rank0_only', True)
        self.debug_check_finite = getattr(config, 'debug_check_finite', False)
        self._debug_forward_count = 0

        text_input_dim = getattr(config, 'text_input_dim', 768)
        mesh_encoder_dim = getattr(config, 'mesh_encoder_dim', 768)
        mesh_feat_dim = getattr(config, 'mesh_feat_dim', 1024)
        pointnet_output_dim = getattr(config, 'pointnet_output_dim', 1024)
        global_dim = getattr(config, 'global_dim', pointnet_output_dim)
        self.use_spatial_pe = getattr(config, 'use_spatial_pe', True)
        self.spatial_num_frequencies = getattr(config, 'spatial_num_frequencies', 10)
        text_depth = getattr(config, 'text_depth', 2)
        text_num_heads = getattr(config, 'text_num_heads', 8)
        local_depth = getattr(config, 'mesh_depth', 1)
        pointnet_depth = getattr(config, 'pointnet_depth', 1)
        cross_attn_num_heads = getattr(config, 'cross_attn_num_heads', 8)
        self_attn_num_heads = getattr(config, 'self_attn_num_heads', 8)
        mesh_num_heads = getattr(config, 'mesh_num_heads', 8)
        pointnet_num_heads = getattr(config, 'pointnet_num_heads', 8)
        deep_fusion_depth = int(getattr(config, 'deep_fusion_depth', 1))
        attn_dropout = getattr(config, 'attn_dropout', getattr(config, 'cross_attn_dropout', 0.1))
        final_dim = getattr(config, 'final_dim', 1024)
        if deep_fusion_depth < 1:
            raise ValueError(f"deep_fusion_depth must be >= 1, got {deep_fusion_depth}")

        # === Branch 1: Text Encoder ===
        self.text_input_dim = text_input_dim
        self.text_dim = getattr(config, 'text_dim', text_input_dim)
        self.text_num_heads = text_num_heads
        self.text_embedding = nn.Linear(text_input_dim, self.text_dim)
        self.text_blocks = nn.ModuleList([
            Block(self.text_dim, self.text_num_heads, qkv_bias=True)
            for _ in range(text_depth)
        ])

        # === Branch 2: Local Mesh Encoder ===
        self.encoder = Mesh_encoder()
        self.encoder.requires_grad_(False)
        # Project Mesh_encoder output (768) to target dimension
        self.mesh_output_proj = nn.Linear(mesh_encoder_dim, mesh_feat_dim)
        self.mesh_feat_dim = mesh_feat_dim
        self.mesh_num_heads = mesh_num_heads
        self.blocks = nn.ModuleList([
            Block(self.mesh_feat_dim, self.mesh_num_heads, qkv_bias=True)
            for _ in range(local_depth)
        ])

        # === Branch 3: Global PointNet Encoder ===
        self.global_encoder = PointNetEncoder()
        self.global_dim = global_dim
        self.pointnet_num_heads = pointnet_num_heads
        self.global_blocks = nn.ModuleList([
            Block(self.global_dim, self.pointnet_num_heads, qkv_bias=True)
            for _ in range(pointnet_depth)
        ])
        self.cross_attn_num_heads = cross_attn_num_heads
        self.self_attn_num_heads = self_attn_num_heads
        self.deep_fusion_depth = deep_fusion_depth

        if self.text_dim % self.cross_attn_num_heads != 0:
            raise ValueError(
                f"text_dim ({self.text_dim}) must be divisible by cross_attn_num_heads ({self.cross_attn_num_heads})"
            )
        if self.text_dim % self.text_num_heads != 0:
            raise ValueError(
                f"text_dim ({self.text_dim}) must be divisible by text_num_heads ({self.text_num_heads})"
            )
        if self.mesh_feat_dim % self.mesh_num_heads != 0:
            raise ValueError(
                f"mesh_feat_dim ({self.mesh_feat_dim}) must be divisible by mesh_num_heads ({self.mesh_num_heads})"
            )
        if self.mesh_feat_dim % self.cross_attn_num_heads != 0:
            raise ValueError(
                f"mesh_feat_dim ({self.mesh_feat_dim}) must be divisible by "
                f"cross_attn_num_heads ({self.cross_attn_num_heads})"
            )
        if self.global_dim % self.cross_attn_num_heads != 0:
            raise ValueError(
                f"global_dim ({self.global_dim}) must be divisible by "
                f"cross_attn_num_heads ({self.cross_attn_num_heads})"
            )
        if self.global_dim % self.pointnet_num_heads != 0:
            raise ValueError(
                f"global_dim ({self.global_dim}) must be divisible by "
                f"pointnet_num_heads ({self.pointnet_num_heads})"
            )
        if (2 * self.text_dim) % self.self_attn_num_heads != 0:
            raise ValueError(
                f"2 * text_dim ({2 * self.text_dim}) must be divisible by "
                f"self_attn_num_heads ({self.self_attn_num_heads})"
            )
        if (2 * self.mesh_feat_dim) % self.self_attn_num_heads != 0:
            raise ValueError(
                f"2 * mesh_feat_dim ({2 * self.mesh_feat_dim}) must be divisible by "
                f"self_attn_num_heads ({self.self_attn_num_heads})"
            )
        if (2 * self.global_dim) % self.self_attn_num_heads != 0:
            raise ValueError(
                f"2 * global_dim ({2 * self.global_dim}) must be divisible by "
                f"self_attn_num_heads ({self.self_attn_num_heads})"
            )

        # === Feature Projection Layers (before Cross-Attention) ===
        self.text_feat_proj = nn.Linear(self.text_dim, self.text_dim)
        self.mesh_feat_proj = nn.Linear(self.mesh_feat_dim, self.mesh_feat_dim)
        self.pointnet_feat_proj = nn.Linear(pointnet_output_dim, self.global_dim)
        self.text_feat_norm = nn.LayerNorm(self.text_dim)
        self.mesh_feat_norm = nn.LayerNorm(self.mesh_feat_dim)
        self.pointnet_feat_norm = nn.LayerNorm(self.global_dim)

        # === Position Embedding for Stage-1 Branch Features ===
        default_slot_to_fdi = [
            18, 17, 16, 15, 14, 13, 12, 11,
            21, 22, 23, 24, 25, 26, 27, 28,
            48, 47, 46, 45, 44, 43, 42, 41,
            31, 32, 33, 34, 35, 36, 37, 38,
        ]
        slot_to_fdi = getattr(config, 'slot_to_fdi', default_slot_to_fdi)
        self.text_pos_pe = FDISemanticPositionEmbedding(embed_dim=self.text_dim, slot_to_fdi=slot_to_fdi)
        self.mesh_pos_pe = FDISemanticPositionEmbedding(embed_dim=self.mesh_feat_dim, slot_to_fdi=slot_to_fdi)
        self.pointnet_pos_pe = FDISemanticPositionEmbedding(embed_dim=self.global_dim, slot_to_fdi=slot_to_fdi)
        if self.use_spatial_pe:
            self.text_spatial_pe = SpatialSinusoidalPositionEmbedding(
                embed_dim=self.text_dim,
                num_frequencies=self.spatial_num_frequencies,
            )
            self.mesh_spatial_pe = SpatialSinusoidalPositionEmbedding(
                embed_dim=self.mesh_feat_dim,
                num_frequencies=self.spatial_num_frequencies,
            )
            self.pointnet_spatial_pe = SpatialSinusoidalPositionEmbedding(
                embed_dim=self.global_dim,
                num_frequencies=self.spatial_num_frequencies,
            )
        else:
            self.text_spatial_pe = None
            self.mesh_spatial_pe = None
            self.pointnet_spatial_pe = None

        # === Cross-Attention: Pairwise Interactions (6 pairs) ===
        # Convention: main modality is K/V, secondary modality is Q.
        # Output always follows secondary modality dimension (embed_dim).

        # Secondary Text branch: (Mesh -> Text), (PointNet -> Text)
        self.text_cross_mesh = CrossAttention(
            embed_dim=self.text_dim,       # Q: Text
            key_dim=self.mesh_feat_dim,    # K/V: Mesh
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )
        self.text_cross_pointnet = CrossAttention(
            embed_dim=self.text_dim,       # Q: Text
            key_dim=self.global_dim,       # K/V: PointNet
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )

        # Secondary Mesh branch: (Text -> Mesh), (PointNet -> Mesh)
        self.mesh_cross_text = CrossAttention(
            embed_dim=self.mesh_feat_dim,  # Q: Mesh
            key_dim=self.text_dim,         # K/V: Text
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )
        self.mesh_cross_pointnet = CrossAttention(
            embed_dim=self.mesh_feat_dim,  # Q: Mesh
            key_dim=self.global_dim,       # K/V: PointNet
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )

        # Secondary PointNet branch: (Text -> PointNet), (Mesh -> PointNet)
        self.pointnet_cross_text = CrossAttention(
            embed_dim=self.global_dim,     # Q: PointNet
            key_dim=self.text_dim,         # K/V: Text
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )
        self.pointnet_cross_mesh = CrossAttention(
            embed_dim=self.global_dim,     # Q: PointNet
            key_dim=self.mesh_feat_dim,    # K/V: Mesh
            num_heads=self.cross_attn_num_heads,
            dropout=attn_dropout
        )

        # === Self-Attention: Intra-modal Enhancement (3 branches) ===
        self.text_self = SelfAttentionStack(
            embed_dim=2 * self.text_dim,  # Concat 2 sources
            depth=self.deep_fusion_depth,
            num_heads=self.self_attn_num_heads,
            dropout=attn_dropout
        )
        self.mesh_self = SelfAttentionStack(
            embed_dim=2 * self.mesh_feat_dim,  # Concat 2 sources
            depth=self.deep_fusion_depth,
            num_heads=self.self_attn_num_heads,
            dropout=attn_dropout
        )
        self.pointnet_self = SelfAttentionStack(
            embed_dim=2 * self.global_dim,  # Concat 2 sources
            depth=self.deep_fusion_depth,
            num_heads=self.self_attn_num_heads,
            dropout=attn_dropout
        )

        # === Final Fusion ===
        fused_dim = 2 * self.text_dim + 2 * self.mesh_feat_dim + 2 * self.global_dim
        # 2*1024 + 2*1024 + 2*1024 = 2048 + 2048 + 2048 = 6144

        self.final_dim = final_dim
        self.fusion_proj = nn.Linear(fused_dim, self.final_dim)
        self.fusion_ln = nn.LayerNorm(self.final_dim)

        # === Direct MLP Regressor ===
        self.regressor = DirectRegressorMLP(in_features=self.final_dim, out_dim=9)
        self.initialze_weights()
        if config.args.encoder_ckpts != '':
            checkpoint = torch.load(config.args.encoder_ckpts)['base_model']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            self.encoder.load_state_dict(new_state_dict, strict=False)
    
    def initialze_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)

    def _debug_on_this_rank(self):
        if not self.debug_rank0_only:
            return True
        rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))
        try:
            return int(rank) == 0
        except ValueError:
            return rank == "0"

    def _should_log_debug(self):
        if not self.debug:
            return False
        if self.debug_max_forwards <= 0:
            return False
        return self._debug_on_this_rank() and self._debug_forward_count < self.debug_max_forwards

    @staticmethod
    def _tensor_stats_str(tensor):
        if tensor is None:
            return "None"
        if not torch.is_tensor(tensor):
            return f"type={type(tensor).__name__}"
        if tensor.numel() == 0:
            return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} numel=0"

        t = tensor.detach()
        tf = t.float() if not t.is_floating_point() else t
        finite_mask = torch.isfinite(tf)
        finite_count = int(finite_mask.sum().item())
        total = tf.numel()
        nan_count = int(torch.isnan(tf).sum().item())
        inf_count = int(torch.isinf(tf).sum().item())

        if finite_count > 0:
            finite_vals = tf[finite_mask]
            min_v = float(finite_vals.min().item())
            max_v = float(finite_vals.max().item())
            mean_v = float(finite_vals.mean().item())
            std_v = float(finite_vals.std(unbiased=False).item())
            absmax_v = float(finite_vals.abs().max().item())
            stats = (
                f"shape={tuple(t.shape)} dtype={t.dtype} "
                f"min={min_v:.6g} max={max_v:.6g} mean={mean_v:.6g} std={std_v:.6g} absmax={absmax_v:.6g}"
            )
        else:
            stats = f"shape={tuple(t.shape)} dtype={t.dtype} all_non_finite=True"

        return f"{stats} finite={finite_count}/{total} nan={nan_count} inf={inf_count}"

    def _debug_tensor(self, name, tensor, debug_enabled):
        if not debug_enabled or not self.debug_stats:
            return
        print(f"[SSMT][STAT] {name:<24} {self._tensor_stats_str(tensor)}")
        if self.debug_check_finite and torch.is_tensor(tensor):
            if not torch.isfinite(tensor.float()).all():
                raise FloatingPointError(f"[SSMT] Non-finite value detected in tensor: {name}")
        
    def forward(self,faces, feats, centers, Fs, cordinates, centroid, points, text_feature, gt_6dof=None, tooth_indices=None):
        '''
        Args:
            faces: [bs,32,256,64,3] - Mesh faces
            feats: [bs,32,10,256,64] - Mesh features
            centers: [bs,32,256,64,3] - Mesh centers
            Fs: [bs,32] - Number of faces per tooth
            cordinates: [bs,32,256,64,9] - Mesh coordinates
            centroid: [bs,32,3] - Physical centroid coordinates used for sinusoidal spatial PE (when enabled)
            points: [bs,32,2048,3] - Point clouds
            text_feature: [bs,10,text_input_dim] - PubMedBERT embedding
            gt_6dof: [bs,32,9] - Ground truth 6DoF (optional, for loss calculation)
            tooth_indices: [bs,32] - Optional tooth slot indices for semantic position embedding
        Returns:
            dofs: [bs,32,9] - 6DoF parameters (3 translation + 6 rotation)
            patient_feats: dict - intermediate features for NCE
        '''
        debug_enabled = self._should_log_debug()
        if debug_enabled:
            print(f"\n[SSMT][Forward {self._debug_forward_count + 1}/{self.debug_max_forwards}] {'=' * 60}")
            self._debug_tensor("input/faces", faces, debug_enabled)
            self._debug_tensor("input/feats", feats, debug_enabled)
            self._debug_tensor("input/centers", centers, debug_enabled)
            self._debug_tensor("input/Fs", Fs, debug_enabled)
            self._debug_tensor("input/cordinates", cordinates, debug_enabled)
            self._debug_tensor("input/centroid", centroid, debug_enabled)
            self._debug_tensor("input/points", points, debug_enabled)
            self._debug_tensor("input/text_feature", text_feature, debug_enabled)
            self._debug_tensor("input/gt_6dof", gt_6dof, debug_enabled)

        n = faces.shape[1]

        # === Branch 1: Text Encoder ===
        text_emb = self.text_embedding(text_feature)  # [bs,10,text_dim]
        text_emb = F.adaptive_avg_pool1d(text_emb.transpose(1, 2), n).transpose(1, 2)  # [bs,n,text_dim]
        for blk in self.text_blocks:
            text_emb = blk(text_emb)
        if debug_enabled:
            print(f"[1] Text branch output: {text_emb.shape}")
        self._debug_tensor("stage1/text_emb", text_emb, debug_enabled)

        # === Branch 2: Local Mesh Encoder ===
        encodings = []
        for i in range(n):
            encoding = self.encoder(faces[:,i],feats[:,i],centers[:,i],Fs[:,i],cordinates[:,i])
            encodings.append(encoding)
        mesh_emb = torch.stack(encodings,dim=1)  # [bs,32,768] (mesh_encoder_dim)
        local_mesh_feat = self.mesh_output_proj(mesh_emb)  # [bs,32,mesh_feat_dim]
        for blk in self.blocks:
            local_mesh_feat = blk(local_mesh_feat)
        if debug_enabled:
            print(f"[2] Mesh features: local={local_mesh_feat.shape}")
        self._debug_tensor("stage1/mesh_emb", mesh_emb, debug_enabled)
        self._debug_tensor("stage1/local_mesh_feat", local_mesh_feat, debug_enabled)

        # === Branch 3: Global PointNet Encoder ===
        global_points = rearrange(points,'b n p c -> b c (n p)')
        global_emb = self.global_encoder(global_points).unsqueeze(1).repeat(1,n,1)  # [bs,32,1024]
        for blk in self.global_blocks:
            global_emb = blk(global_emb)
        if debug_enabled:
            print(f"[3] PointNet features: global={global_emb.shape}")
        self._debug_tensor("stage1/global_points", global_points, debug_enabled)
        self._debug_tensor("stage1/global_emb", global_emb, debug_enabled)

        # === Feature Projection ===
        text_emb = self.text_feat_proj(text_emb)  # [bs,32,text_dim]
        local_mesh_feat = self.mesh_feat_proj(local_mesh_feat)  # [bs,32,mesh_feat_dim]
        global_emb = self.pointnet_feat_proj(global_emb)  # [bs,32,global_dim]
        text_emb = self.text_feat_norm(text_emb)
        local_mesh_feat = self.mesh_feat_norm(local_mesh_feat)
        global_emb = self.pointnet_feat_norm(global_emb)
        self._debug_tensor("stage1/text_proj+norm", text_emb, debug_enabled)
        self._debug_tensor("stage1/mesh_proj+norm", local_mesh_feat, debug_enabled)
        self._debug_tensor("stage1/pointnet_proj+norm", global_emb, debug_enabled)
        if tooth_indices is None:
            bs = text_emb.shape[0]
            tooth_indices = torch.arange(n, dtype=torch.long, device=text_emb.device).unsqueeze(0).expand(bs, -1)
        else:
            tooth_indices = tooth_indices.to(device=text_emb.device, dtype=torch.long)

        # Position embeddings are injected here for all three Stage-1 branches:
        # 1) FDI semantic PE, 2) physical 3D sinusoidal PE.
        text_emb = self.text_pos_pe(text_emb, tooth_indices)
        local_mesh_feat = self.mesh_pos_pe(local_mesh_feat, tooth_indices)
        global_emb = self.pointnet_pos_pe(global_emb, tooth_indices)
        if self.use_spatial_pe:
            centroid = centroid.to(device=text_emb.device)
            text_emb = self.text_spatial_pe(centroids=centroid, x=text_emb)
            local_mesh_feat = self.mesh_spatial_pe(centroids=centroid, x=local_mesh_feat)
            global_emb = self.pointnet_spatial_pe(centroids=centroid, x=global_emb)
        if debug_enabled:
            print(f"[4] After projection: text={text_emb.shape}, mesh={local_mesh_feat.shape}, pointnet={global_emb.shape}")
        self._debug_tensor("stage1/text_proj+pe", text_emb, debug_enabled)
        self._debug_tensor("stage1/mesh_proj+pe", local_mesh_feat, debug_enabled)
        self._debug_tensor("stage1/pointnet_proj+pe", global_emb, debug_enabled)

        # === Stage 1: Cross-Modal Attention ===
        # Text branch
        text_from_mesh = self.text_cross_mesh(local_mesh_feat, text_emb)
        text_from_pointnet = self.text_cross_pointnet(global_emb, text_emb)
        text_fused = torch.cat([text_from_mesh, text_from_pointnet], dim=-1)  # [bs,n,2*text_dim]

        # Mesh branch
        mesh_from_text = self.mesh_cross_text(text_emb, local_mesh_feat)
        mesh_from_pointnet = self.mesh_cross_pointnet(global_emb, local_mesh_feat)
        mesh_fused = torch.cat([mesh_from_text, mesh_from_pointnet], dim=-1)  # [bs,32,2048]

        # PointNet branch
        pointnet_from_text = self.pointnet_cross_text(text_emb, global_emb)
        pointnet_from_mesh = self.pointnet_cross_mesh(local_mesh_feat, global_emb)
        pointnet_fused = torch.cat([pointnet_from_text, pointnet_from_mesh], dim=-1)  # [bs,32,2048]
        if debug_enabled:
            print(f"[5] Cross-Attention: text={text_fused.shape}, mesh={mesh_fused.shape}, pointnet={pointnet_fused.shape}")
        self._debug_tensor("stage2/text_fused", text_fused, debug_enabled)
        self._debug_tensor("stage2/mesh_fused", mesh_fused, debug_enabled)
        self._debug_tensor("stage2/pointnet_fused", pointnet_fused, debug_enabled)

        # === Stage 2: Self-Attention ===
        text_enhanced = self.text_self(text_fused)  # [bs,32,2048]
        mesh_enhanced = self.mesh_self(mesh_fused)  # [bs,32,2048]
        pointnet_enhanced = self.pointnet_self(pointnet_fused)  # [bs,32,2048]
        if debug_enabled:
            print(f"[6] Self-Attention: text={text_enhanced.shape}, mesh={mesh_enhanced.shape}, pointnet={pointnet_enhanced.shape}")
        self._debug_tensor("stage3/text_enhanced", text_enhanced, debug_enabled)
        self._debug_tensor("stage3/mesh_enhanced", mesh_enhanced, debug_enabled)
        self._debug_tensor("stage3/pointnet_enhanced", pointnet_enhanced, debug_enabled)

        # === Stage 3: Final Fusion ===
        final_feat = torch.cat([text_enhanced, mesh_enhanced, pointnet_enhanced], dim=-1)  # [bs,32,6144]
        self._debug_tensor("stage4/final_feat_cat", final_feat, debug_enabled)
        final_feat = self.fusion_proj(final_feat)  # [bs,32,final_dim]
        self._debug_tensor("stage4/final_feat_proj", final_feat, debug_enabled)
        final_feat = self.fusion_ln(final_feat)
        self._debug_tensor("stage4/final_feat_ln", final_feat, debug_enabled)
        if debug_enabled:
            print(f"[7] Final fusion: {final_feat.shape}")

        # === Stage 4: Regression ===
        dofs = self.regressor(final_feat)  # [bs,32,9] - Direct MLP prediction
        self._debug_tensor("stage5/dofs", dofs, debug_enabled)
        if debug_enabled:
            print(f"[8] 6DoF output: {dofs.shape}")
            print(f"{'='*60}\n")

        if debug_enabled:
            self._debug_forward_count += 1

        # Intermediate features for contrastive learning (NCE).
        patient_feats = {
            'local_mesh_feat': local_mesh_feat,
            'global_pointcloud_feat': global_emb,
            'text_feat': text_emb,
            'fused_feat': final_feat,
        }
        return dofs.float(), patient_feats
