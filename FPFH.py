import open3d as o3d
import open3d.core as o3c
import torch
from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import knn_points,knn_gather
from knn_cuda import KNN
import numpy as np


def batched_knn(knn, reference, query, batch_size=4000):
    all_idx = []
    for i in range(0, query.shape[1], batch_size):
        q_batch = query[:, i:i+batch_size, :]  # shape [B, b, 3]
        _, idx = knn(reference, q_batch)    # shape [B, b, k]
        all_idx.append(idx)
    return torch.cat(all_idx, dim=1)           # [B, G, k]


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class Multi_FPFHFeatures:
    def __init__(
        self,
        device="cuda:0",
        token_num_base=1024,  # l4 points: 2*token_num_base
    ):

        self.device = device
        self.token_num_base = token_num_base

        self.radius_normal = 10000 #10000
        self.radius_fpfh = 1000000 #1000000 

    def _multi_scale_fps_random_first(self, pc_b, n1, n2, n3, n4):
        B, N, _ = pc_b.shape
        device = pc_b.device

        assert B == 1, "current implementation only supports batch size = 1"

        # 随机选一个点，与第0个点交换
        j = torch.randint(0, N, (1,), device=device).item()

        if j == 0:
            pc_b_mod = pc_b
        else:
            pc_b_mod = pc_b.clone()
            pc_b_mod[:, [0, j], :] = pc_b_mod[:, [j, 0], :]

        idx_all_int = pointnet2_utils.furthest_point_sample(pc_b_mod, n1)

        xyz_all = pointnet2_utils.gather_operation(
            pc_b_mod.transpose(1, 2).contiguous(), idx_all_int
        ).transpose(1, 2)

        idx_all_long = idx_all_int.long()[0]

        # 把修改后的索引映射回原始点云索引
        idx_all_original = idx_all_long.clone()
        if j != 0:
            idx_all_original[idx_all_long == 0] = j
            idx_all_original[idx_all_long == j] = 0

        xyz_l1, idx_l1 = xyz_all[:, :n1], idx_all_original[:n1]
        xyz_l2, idx_l2 = xyz_all[:, :n2], idx_all_original[:n2]
        xyz_l3, idx_l3 = xyz_all[:, :n3], idx_all_original[:n3]
        xyz_l4, idx_l4 = xyz_all[:, :n4], idx_all_original[:n4]

        return (
            xyz_l1[0], idx_l1.long(),
            xyz_l2[0], idx_l2.long(),
            xyz_l3[0], idx_l3.long(),
            xyz_l4[0], idx_l4.long(),
        )
    def _multi_scale_fps_random(self, pc_b, n1, n2, n3, n4):
        B, N, _ = pc_b.shape
        device = pc_b.device

        assert B == 1, "current implementation only supports batch size = 1"

        # 随机打乱输入点顺序
        rand_perm = torch.randperm(N, device=device)   # int64
        pc_b_shuffled = pc_b[:, rand_perm, :]

        # FPS输出保留为int32，给pointnet2的gather_operation使用
        idx_all_int = pointnet2_utils.furthest_point_sample(pc_b_shuffled, n1)

        xyz_all = pointnet2_utils.gather_operation(
            pc_b_shuffled.transpose(1, 2).contiguous(),
            idx_all_int
        ).transpose(1, 2)

        # 另存一份long，用于PyTorch索引
        idx_all_long = idx_all_int.long()

        # 映射回原始点云中的索引
        idx_all_original = rand_perm[idx_all_long[0]]

        xyz_l1, idx_l1 = xyz_all[:, :n1], idx_all_original[:n1]
        xyz_l2, idx_l2 = xyz_all[:, :n2], idx_all_original[:n2]
        xyz_l3, idx_l3 = xyz_all[:, :n3], idx_all_original[:n3]
        xyz_l4, idx_l4 = xyz_all[:, :n4], idx_all_original[:n4]

        return (
            xyz_l1[0], idx_l1.long(),
            xyz_l2[0], idx_l2.long(),
            xyz_l3[0], idx_l3.long(),
            xyz_l4[0], idx_l4.long(),
        )

    def _multi_scale_fps(self, pc_b, n1, n2, n3, n4):

        idx_all = pointnet2_utils.furthest_point_sample(pc_b, n1)

        xyz_all = pointnet2_utils.gather_operation(
            pc_b.transpose(1, 2).contiguous(), idx_all
        ).transpose(1, 2)

        xyz_l1, idx_l1 = xyz_all, idx_all
        xyz_l2, idx_l2 = xyz_all[:, :n2], idx_all[:, :n2]
        xyz_l3, idx_l3 = xyz_all[:, :n3], idx_all[:, :n3]
        xyz_l4, idx_l4 = xyz_all[:, :n4], idx_all[:, :n4]

        return (
            xyz_l1[0], idx_l1[0].long(),
            xyz_l2[0], idx_l2[0].long(),
            xyz_l3[0], idx_l3[0].long(),
            xyz_l4[0], idx_l4[0].long(),
        )
    def _estimate_and_orient_normals2(self, pc, pc_cuda, torch_dev,o3d_dev,orient):

        
        o3d_pc = o3d.geometry.PointCloud()

        o3d_pc.points = o3d.utility.Vector3dVector(pc.cpu().numpy())



        o3d_pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.radius_normal,
            max_nn=10,
        ))
        


        normals = torch.from_numpy(

            np.asarray(o3d_pc.normals, dtype=np.float32)

        ).to(torch_dev, non_blocking=True)


        if orient:
            center = pc_cuda.mean(dim=0, keepdim=True)
            dot = ((pc_cuda - center) * normals).sum(dim=-1, keepdim=True)
            normals[dot.squeeze(-1) < 0] *= -1
        
        return normals
            
        
    def _estimate_and_orient_normals(self, pc, o3d_dev, orient):

        
        pcd = o3d.t.geometry.PointCloud(o3d_dev)
        pcd.point["positions"] = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(pc)
        )

        pcd.estimate_normals(radius=self.radius_normal, max_nn=10)

        normals = torch.utils.dlpack.from_dlpack(
            pcd.point["normals"].to_dlpack()
        ).to(torch.float32)

        if orient:
            center = pc.mean(dim=0, keepdim=True)
            dot = ((pc - center) * normals).sum(dim=-1, keepdim=True)
            normals[dot.squeeze(-1) < 0] *= -1
        

        return normals

    def _compute_fpfh(self, xyz, normals, o3d_dev, radius, max_nn):
        pcd = o3d.t.geometry.PointCloud(o3d_dev)
        pcd.point["positions"] = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(xyz)
        )
        pcd.point["normals"] = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(normals)
        )

        fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
            pcd,
            radius=radius,
            max_nn=max_nn,
        )

        feat = torch.utils.dlpack.from_dlpack(
            fpfh.to_dlpack()
        ).to(torch.float32)

        xyz_out = torch.utils.dlpack.from_dlpack(
            pcd.point["positions"].to_dlpack()
        ).to(torch.float32)

        return xyz_out, feat

    def _aggregate(self, src_xyz, dst_xyz, src_feat, group_size):
        src_xyz_b = src_xyz.unsqueeze(0)
        dst_xyz_b = dst_xyz.unsqueeze(0)
        feat_b = src_feat.unsqueeze(0).transpose(1, 2).contiguous()

        _, idx, _ = knn_points(dst_xyz_b, src_xyz_b,
                               K=group_size, return_nn=False)
        
        # grouped_feat = knn_gather(feat_b, idx)
        # agg = grouped_feat.mean(dim=2)
        idx = idx.int()
        grouped = pointnet2_utils.grouping_operation(feat_b, idx)
        agg = grouped.mean(dim=-1).transpose(1, 2)

        return agg[0]

    def get_fpfh_features(self, unorganized_pc, orient=True,randomFPS=False):

        dev = torch.device(self.device)

        unorganized_pc=unorganized_pc.squeeze()
        pc = torch.as_tensor(unorganized_pc, dtype=torch.float32, device=dev)


        N = pc.shape[0]
        pc_b = pc.unsqueeze(0)

        num_b = self.token_num_base  # anomalyShapenet  1024, mulsen 2048
        n1 = min(N, num_b * 12)
        n2 = min(N, num_b * 8)
        n3 = min(N, num_b * 4)
        n4 = min(N, num_b * 2)

        
        if randomFPS:
            (
                xyz_l1, idx_l1,
                xyz_l2, idx_l2,
                xyz_l3, idx_l3,
                xyz_l4, idx_l4,
            ) = self._multi_scale_fps_random_first(pc_b, n1, n2, n3, n4)
        else:
            (
                xyz_l1, idx_l1,
                xyz_l2, idx_l2,
                xyz_l3, idx_l3,
                xyz_l4, idx_l4,
            ) = self._multi_scale_fps(pc_b, n1, n2, n3, n4)
        o3d_dev = o3d.core.Device(str(dev).upper())

        normals = self._estimate_and_orient_normals(pc, o3d_dev, orient=orient)
        # normals = self._estimate_and_orient_normals2(unorganized_pc, pc,dev,o3d_dev,orient=orient)

        nrm_l1 = normals[idx_l1]
        nrm_l2 = normals[idx_l2]
        nrm_l3 = normals[idx_l3]
        nrm_l4 = normals[idx_l4]



        # normals_l1 = self._estimate_and_orient_normals2(

        #     xyz_l1, xyz_l1, dev, o3d_dev, orient=orient

        # )

        # nrm_l1 = normals_l1

        # nrm_l2 = normals_l1[:n2]

        # nrm_l3 = normals_l1[:n3]

        # nrm_l4 = normals_l1[:n4]
        xyz_l1, feat_l1 = self._compute_fpfh(
            xyz_l1, nrm_l1, o3d_dev, radius=self.radius_fpfh, max_nn=80)
        xyz_l2, feat_l2 = self._compute_fpfh(
            xyz_l2, nrm_l2, o3d_dev, radius=self.radius_fpfh, max_nn=80)
        
        xyz_l3, feat_l3 = self._compute_fpfh(
            xyz_l3, nrm_l3, o3d_dev, radius=self.radius_fpfh, max_nn=80)
        xyz_l4, feat_l4 = self._compute_fpfh(
            xyz_l4, nrm_l4, o3d_dev, radius=self.radius_fpfh, max_nn=80)

        feat_l1_to_l4 = self._aggregate(
            xyz_l1, xyz_l4, feat_l1, group_size=256)
        feat_l2_to_l4 = self._aggregate(
            xyz_l2, xyz_l4, feat_l2, group_size=214)
        feat_l3_to_l4 = self._aggregate(
            xyz_l3, xyz_l4, feat_l3, group_size=172)
        feat_l4_local = self._aggregate(
            xyz_l4, xyz_l4, feat_l4, group_size=128)

        final_feat = torch.cat(
            [feat_l1_to_l4, feat_l2_to_l4, feat_l3_to_l4, feat_l4_local],
            dim=-1,
        )



        return {
            "fpfhs": final_feat,   # (N4, C)
            "centers": xyz_l4,     # (N4, 3)
            "pc": pc,
        }
