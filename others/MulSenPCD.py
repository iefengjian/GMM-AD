import pathlib
import csv
from torch.utils.data import Dataset
import glob
import os
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader
import re
import csv
from scipy.spatial import KDTree
import torch

import random
from dataset_3D import DatasetMulSen_ad_train, DatasetMulSen_ad_test



def mulsen_classes():
    return [
        "capsule",
        "cotton",
        "cube",
        "spring_pad",
        "screw",
        "screen",
        "piggy",
        "nut",
        "flat_pad",
        'plastic_cylinder',
        "zipper",
        "button_cell",
        "toothbrush",
        "solar_panel",
        "light",
    ]


from pathlib import Path

def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_pcd(p: Path, pc:np.ndarray):
    _safe_mkdir(p.parent)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pc)
    ok = o3d.io.write_point_cloud(str(p), source, write_ascii=True, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write pcd: {p}")
    
if __name__=="__main__":

    mulsen_path= '../datasets/MulSen_AD'
    save_dir = '../datasets/MulSen_AD_PCD'

    clses = mulsen_classes()
    for cls in clses:
        print(cls)
        cls_dir = Path(save_dir)/cls/"train"
        _safe_mkdir(cls_dir)
        data=DatasetMulSen_ad_train(dataset_dir=mulsen_path,cls_name=cls, num_points=1024,if_norm=True)
        for x in data:
            pointcloud, mask, label, name = x
            name=Path(name)
            new_name=f"{name.stem}.pcd"
            _write_pcd(cls_dir/new_name,pointcloud)
            
    for cls in clses:
        print(cls)
        test_dir = Path(save_dir)/cls/"test"
        _safe_mkdir(test_dir)
        gt_dir = Path(save_dir)/cls/"GT"
        _safe_mkdir(gt_dir)
        data=DatasetMulSen_ad_test(dataset_dir=mulsen_path,cls_name=cls, num_points=1024,if_norm=True)
        for x in data:
            pointcloud, mask, label, name = x
            name=Path(name)
            defect_type=name.parent.name
            new_name=test_dir/ (cls+"_"+defect_type+name.stem+".pcd")
            _write_pcd(test_dir/new_name,pointcloud)
            if label==1:
                gt_txt=gt_dir/ (cls+"_"+defect_type+name.stem+".txt")
                out=np.concatenate([pointcloud,mask.reshape(-1,1)],axis=1)
                _safe_mkdir(gt_txt.parent)
                np.savetxt(gt_txt, out, fmt="%.8f", delimiter=",")





