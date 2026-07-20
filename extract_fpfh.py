import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from pathlib import Path


from FPFH import Multi_FPFHFeatures
from utils import center_shift



class PCDTrain(Dataset): # anomalyshapenet mulsen
    def __init__(self, class_name,dataset_path,):

        self.cls = class_name
        self.pcd_path = os.path.join(dataset_path, self.cls, "train")

        self.dataset_path = dataset_path
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []
        pcd_paths = glob.glob(self.pcd_path+ "/*.pcd")
        pcd_paths = sorted(pcd_paths)
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels
    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        points = center_shift(points)
        return points, label, label, pcd_path


class Real3DTrain(Dataset): # real3d
    def __init__(self, class_name,dataset_path):

        self.cls = class_name
        self.pcd_path = os.path.join(dataset_path, self.cls, "train_cut")
        self.dataset_path = dataset_path
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []
        pcd_paths = glob.glob(self.pcd_path+ "/*.asc")
        pcd_paths = sorted(pcd_paths)
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels
    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        points = np.loadtxt(pcd_path,dtype=np.float32)
        points = center_shift(points)
        return points, label, label, pcd_path

class PCDTest(Dataset):
    def __init__(self, class_name,dataset_path):

        self.cls = class_name
        self.pcd_path = os.path.join(dataset_path, self.cls, "test")
        self.dataset_path = dataset_path
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        pcd_paths = glob.glob(self.pcd_path+ "/*.pcd")
        gt_path = glob.glob(self.pcd_path.replace("test",'gt')+ "/*.txt")
        dt="shape" if "ShapeNet" in self.pcd_path else "mulsen"
        if len(gt_path)==0:
            gt_path = glob.glob(self.pcd_path.replace("test",'GT')+ "/*.txt")
        
        for path in pcd_paths:
            if dt=="shape":
                if not "positive" in path:
                    continue
            else:
                if not "good" in path:
                    continue

            pcd_tot_paths.append(path)
        tot_labels.extend([0]*len(pcd_tot_paths))
        pcd_tot_paths.extend(gt_path)
        tot_labels.extend([1]*len(gt_path))


        assert len(pcd_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]

        if label == 0:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points, dtype=np.float32)
            gt = torch.zeros(
                [1, points.shape[0]])
        else:
            try:
                input_points = np.loadtxt(pcd_path, dtype=np.float32, delimiter=' ')
            except ValueError:
                input_points = np.loadtxt(pcd_path, dtype=np.float32, delimiter=',')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(input_points[:,0:3])
            points = np.asarray(pcd.points, dtype=np.float32)

            gt = torch.Tensor(input_points[:,3])


            gt = torch.where(gt > 0.5, 1., .0)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
        
        points = center_shift(points)

        return points, gt[:1], label, pcd_path



def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_train_fpfh(category,dataset_path,out_dir,DS=PCDTrain, randomFPS=False, device="cuda:0",mark=-1): # mark: Filename extension, -1:none

    _safe_mkdir(Path(out_dir)/category)

    dataset = DS(class_name=category,dataset_path=dataset_path)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,pin_memory=True)
    fpfh_extractor = Multi_FPFHFeatures(device=device)

    for pts, mask, label, pcd_path in loader:
        points = pts.to(device)
        data=fpfh_extractor.get_fpfh_features(points,randomFPS=randomFPS)
        fpfhs,centers=data["fpfhs"].squeeze(),data["centers"].squeeze()
        pack = {              
            "pc":pts.squeeze(), # pointclouds
            "fpfhs":fpfhs.squeeze().cpu(),
            "centers":centers.squeeze().cpu(),
        }
        pcd_stem =Path(pcd_path[0]).stem
        if mark==-1:
            out_name = pcd_stem+".pt"
        else:
            out_name = pcd_stem+"_x"+str(mark)+".pt"
        out_path = os.path.join(out_dir,category,out_name)
        torch.save(pack, str(out_path))


def save_test_fpfh(category,dataset_path,out_dir,device="cuda:0"):
    _safe_mkdir(Path(out_dir)/category)
    data = PCDTest(category,dataset_path=dataset_path)
    loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=1, drop_last=False,pin_memory=True)
    
    fpfh_extractor = Multi_FPFHFeatures(device=device)

    for pts, mask, label, pcd_path in loader:

        # points = pts.to(device, dtype=torch.float32)
        data=fpfh_extractor.get_fpfh_features(pts)
        fpfhs,centers=data["fpfhs"].squeeze(),data["centers"].squeeze()

        pack = {              
            "pc":pts.squeeze(),
            "fpfhs":fpfhs.squeeze().cpu(),
            "centers":centers.squeeze().cpu(),
            "mask":mask.squeeze(0).cpu(),
            "label":int(label)
        }
        out_path = os.path.join(out_dir,category,os.path.basename(pcd_path[0]).split(".")[0]+".pt")
        torch.save(pack, str(out_path))


def shapenet3d_classes():
    return [
        "ashtray0", "bag0", "bottle0", "bottle1", "bottle3", "bowl0", "bowl1", "bowl2", "bowl3", "bowl4",
        "bowl5", "bucket0", "bucket1", "cap0", "cap3", "cap4", "cap5", "cup0", "cup1", "eraser0", "headset0", 
        "headset1", "helmet0", "helmet1", "helmet2", "helmet3", "jar0", "microphone0", "shelf0", "tap0", 
        "tap1", "vase0", "vase1", "vase2", "vase3", "vase4", "vase5", "vase7", "vase8", "vase9",
    ]

def mulsen_classes():
    return [
        "capsule", "cotton", "cube", "spring_pad", "screw", "screen", "piggy", "nut", "flat_pad",
        'plastic_cylinder', "zipper", "button_cell", "toothbrush", "solar_panel", "light",
    ]

def real3d_classes():
    return ['airplane','car','candybar','chicken', 'diamond','duck','fish','gemstone',
            'seahorse','shell','starfish','toffees']


if __name__ == "__main__":


    classes = shapenet3d_classes()
    pcd_path = "datasets/Anomaly-ShapeNet-v2/dataset/pcd"
    out_dir = Path("datasets/Shape3D_fpfh")
    for cls in classes:
        print(f"extrach train fpfh {cls}")
        for times in range(10): # shapenet3d data x10 to enlarge the dataset
            save_train_fpfh(cls,pcd_path,out_dir/"train",DS=PCDTrain,randomFPS=True,mark=str(times))
    for cls in classes:
        print(f"extrach test fpfh {cls}")
        save_test_fpfh(cls,pcd_path,out_dir/"test")



    classes = real3d_classes()
    pcd_path_train = "datasets/Real3d_cut"
    pcd_path_test = "datasets/Real3D-AD-PCD"
    out_dir = Path("datasets/Real3D_fpfh")
    for cls in classes:
        print(f"extrach train fpfh {cls}")
        for times in range(10): # shapenet3d data x10 to enlarge the dataset
            save_train_fpfh(cls,pcd_path_train,out_dir/"train",DS=Real3DTrain,randomFPS=True, mark=str(times))
    for cls in classes:
        print(f"extrach test fpfh {cls}")
        save_test_fpfh(cls,pcd_path_test,out_dir/"test")


    classes = mulsen_classes()
    pcd_path = "datasets/MulSen_AD_PCD" #/home/lpw/fengjian/code/tmp/datasets/MulSen_AD_PCD"
    out_dir = Path("datasets/Mulsen_fpfh")
    for cls in classes:
        print(f"extrach train fpfh {cls}")
        save_train_fpfh(cls,pcd_path,out_dir/"train")
    for cls in classes:
        print(f"extrach test fpfh {cls}")
        save_test_fpfh(cls,pcd_path,out_dir/"test")
        


