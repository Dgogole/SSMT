import torch
import torch.nn as nn
from models.pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        # Wider PointNet++ backbone for higher capacity while preserving a 1024-d output.
        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.1, 0.2, 0.4],
            [16, 32, 128],
            0,
            [[64, 64, 128], [128, 128, 256], [128, 192, 256]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4, 0.8],
            [32, 64, 128],
            640,
            [[128, 128, 256], [256, 256, 512], [256, 384, 512]],
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 1280 + 3, [512, 1024, 1024], True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)
        return x


if __name__ == '__main__':
    model = PointNetEncoder()
    xyz = torch.rand(32, 3, 512)
    print(model(xyz).shape)
