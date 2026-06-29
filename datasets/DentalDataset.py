import numpy as np
import os
from collections import OrderedDict
import torch
import torch.utils.data as data
import trimesh
from utils.builder import DATASETS
from utils.pointcloud import read_pointcloud
from utils.rotation import matrix_to_rotation_6d


def load_mesh_shape(mesh, request=[], seed=None):

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # 法向量
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    # 曲率信息
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats).astype(np.float32, copy=False)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.arange(Fs, dtype=np.int32)
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices].astype(np.float32, copy=False)
    center_patch = face_center[indices].astype(np.float32, copy=False)
    cordinates_patch = face_coordinate[indices].astype(np.float32, copy=False)
    faces_patch = mesh.faces[indices].astype(np.int32, copy=False)

    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.int32)), 0)
    Fs_patcha = np.float32(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha

@DATASETS.register_module()
class DentalDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.dataroot = config.dataroot
        self.paramroot = config.paramroot
        self.before_path = config.before_path
        self.after_path = config.after_path
        self.npoint = config.npoint
        self.text_feature_path = config.text_feature_path
        with open(config.file) as f:
            self.indexes = [(i.strip()) for i in f.readlines()]
        self.train = config.train
        self.cache_text_feature = bool(getattr(config, 'cache_text_feature', True))
        self.text_cache_size = max(0, int(getattr(config, 'text_cache_size', 256)))

        self.matrix_paths = [os.path.join(self.paramroot, f'{index}.npy') for index in self.indexes]
        self.text_feature_paths = [os.path.join(self.text_feature_path, f'{index}.pt') for index in self.indexes]
        self.sample_tooth_paths = []
        for index in self.indexes:
            available_teeth = []
            for i in range(32):
                obj_path = os.path.join(self.dataroot, f'{index}_{i}.obj')
                before_path = os.path.join(self.before_path, f'{index}_{i}.ply')
                after_path = os.path.join(self.after_path, f'{index}_{i}.ply')
                if os.path.exists(obj_path) and os.path.exists(before_path) and os.path.exists(after_path):
                    available_teeth.append((i, obj_path, before_path, after_path))
            self.sample_tooth_paths.append(available_teeth)
        self._text_feature_cache = OrderedDict()

    def _load_text_feature(self, idx):
        index = self.indexes[idx]
        if self.cache_text_feature and index in self._text_feature_cache:
            self._text_feature_cache.move_to_end(index)
            return self._text_feature_cache[index]

        text_feat = torch.load(self.text_feature_paths[idx], map_location='cpu')
        if torch.is_tensor(text_feat):
            text_feat = text_feat.detach().cpu().numpy()
        text_feat = np.asarray(text_feat, dtype=np.float32)

        if self.cache_text_feature and self.text_cache_size > 0:
            self._text_feature_cache[index] = text_feat
            self._text_feature_cache.move_to_end(index)
            while len(self._text_feature_cache) > self.text_cache_size:
                self._text_feature_cache.popitem(last=False)
        return text_feat
    
    def __getitem__(self, idx):
        point_num = self.npoint
        feats = np.zeros((32, 10, 256, 64), dtype=np.float32)
        center = np.zeros((32, 256, 64, 3), dtype=np.float32)
        cordinates = np.zeros((32, 256, 64, 9), dtype=np.float32)
        faces = np.zeros((32, 256, 64, 3), dtype=np.int32)
        Fs = np.zeros(32, dtype=np.float32)
        before_points = np.zeros((32, point_num, 3), dtype=np.float32)
        after_points = np.zeros((32, point_num, 3), dtype=np.float32)
        centroid = np.zeros((32, 3), dtype=np.float32)
        after_centroid = np.zeros((32, 3), dtype=np.float32)
        index = self.indexes[idx]
        masks = np.zeros((32,), dtype=np.int32)
        ### get gt rotation and translation params
        matrix = np.load(self.matrix_paths[idx]).astype(np.float32, copy=False)
        # gt_params = torch.cat([torch.from_numpy(after_centroid),rot6d],dim=-1)
        text_feature = np.zeros((10, 768), dtype=np.float32)  # 10个文本token × 768维PubMedBERT特征

        # read 3D features
        for i, obj_path, before_path, after_path in self.sample_tooth_paths[idx]:
            mesh = trimesh.load_mesh(obj_path, process=False)
            masks[i] = 1
            before = read_pointcloud(before_path)
            before_points[i] = before[:point_num]
            centroid[i] = before[point_num]
            after = read_pointcloud(after_path)
            after_points[i] = after[:point_num]
            after_centroid[i] = after[point_num]

            feats[i], center[i], cordinates[i], faces[i], Fs[i] = load_mesh_shape(mesh, request=self.feats)
        # read text features
        # 新格式: [10, 768] - 10个正畸字段 × 768 PubMedBERT维度
        # 处理流程: [10, 768] 直接保留10个token，用于Cross-Attention
        text_feat = self._load_text_feature(idx)  # [10, 768]

        # 直接使用10个文本token（不做平均池化），用于Cross-Attention融合
        text_feature[:] = text_feat  # [10, 768] 

        rot6d = matrix_to_rotation_6d(torch.from_numpy(matrix))
        gt_params = torch.cat([torch.from_numpy(after_centroid), rot6d], dim=-1)
        return index, feats, center, cordinates, faces, Fs, before_points, after_points, centroid, after_centroid, gt_params, masks, text_feature


    def __len__(self):
        return len(self.indexes)
