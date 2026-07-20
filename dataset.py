from torch.utils.data import Dataset
import os
import torch

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


class FPFHData(Dataset):

    def __init__(self, dataset_path, split, class_name):

        self.dataset_path = dataset_path
        self.split = split
        self.cls = class_name
        self.root = os.path.join(dataset_path, split, class_name)
        files = os.listdir(self.root)
        self.files = files
        self._len = len(self.files)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        file = self.files[idx]
        file_path = os.path.join(self.root, file)
        pack = torch.load(file_path, map_location="cpu")
        pack["file"] = file_path
        return pack
