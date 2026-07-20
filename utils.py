import torch
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return torch.clamp(dist, min=1e-6)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def interpolating_points_chunked(xyz1, xyz2, points2, chunk_size=50000):
    """
    Interpolates features from xyz2/points2 to xyz1 using chunked processing to save memory.
    
    Args:
        xyz1: [B, C, N] - Target points (dense)
        xyz2: [B, C, S] - Source points (sparse / center)
        points2: [B, D, S] - Features on source points
        chunk_size: Number of target points to process at once

    Returns:
        interpolated_points: [B, D, N] - Interpolated features on xyz1
    """
    B, C, N = xyz1.shape
    _, _, S = xyz2.shape
    D = points2.shape[1]

    device = xyz1.device
    interpolated_chunks = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xyz1_chunk = xyz1[:, :, start:end]  # [B, C, chunk]
        
        # Permute to [B, chunk, C] for distance computation
        xyz1_chunk_p = xyz1_chunk.permute(0, 2, 1)  # [B, chunk, C]
        xyz2_p = xyz2.permute(0, 2, 1)              # [B, S, C]
        points2_p = points2.permute(0, 2, 1)        # [B, S, D]

        if S == 1:
            interp = points2_p.repeat(1, end - start, 1)  # [B, chunk, D]
        else:
            dists = square_distance(xyz1_chunk_p, xyz2_p)  # [B, chunk, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, chunk, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, chunk, 3]

            interpolated = torch.sum(
                index_points(points2_p, idx) * weight.view(B, -1, 3, 1), dim=2
            )  # [B, chunk, D]
            interp = interpolated  # [B, chunk, D]

        interp = interp.permute(0, 2, 1)  # [B, D, chunk]
        interpolated_chunks.append(interp)

    interpolated_points = torch.cat(interpolated_chunks, dim=2)  # [B, D, N]
    return interpolated_points




def center_shift(points):
    # points: (N,3) or (B,N,3)
    if isinstance(points, torch.Tensor):
        points = points - points.mean(dim=-2, keepdim=True)
    elif isinstance(points, np.ndarray): # numpy
        points = points - points.mean(axis=-2, keepdims=True)
    else:
        raise TypeError(f"Unsupported type: {type(points)}")
    return points

@torch.no_grad()
def UpsampleScores(s_map, points, center,dataset="shapenet"):

    device = points.device

    B = center.shape[0]
    s_map=s_map.reshape(B,1,-1)
    s_map = interpolating_points_chunked(points.permute(0,2,1).to(device), 
                            center.permute(0,2,1).to(device), s_map.to(device)) #B,1,N

    s = torch.max(s_map)

    if dataset == 'real3d':
        s = torch.mean(s_map)
    if dataset == 'shapenet':
        tmp_s,_ = torch.topk(s_map, 80)
        s = torch.mean(tmp_s)
    if dataset == 'mulsen':
        tmp_s,_ = torch.topk(s_map, 80)
        s = torch.mean(tmp_s)    
    return s_map,s  

