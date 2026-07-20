# GMM-AD

##  Environment Requirements

```bash 
python == 3.8
torch == 1.12.1
open3d == 0.19.0 
```
KNN_CUDA

```bash 
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl 
```
Pointnet2_PyTorch

```bash 
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git 
cd Pointnet2_PyTorch 
pip install -r requirements.txt pip install -e . 
cd ..
``` 

## 1. Data Preparation

Please download the following datasets and place them to the ```datasets``` directory:

- Anomaly-ShapeNet-v2
- MulSen_AD
- Real3D-AD-PCD
- Real3d_cut

The original 3D files in the MulSen_AD dataset are not point cloud format. This project uses the official point cloud extraction pipeline. Run the following commands:

```bash 
cd others 
python MulSenPCD.py 
cd .. 
```



## 2. Training and Testing

```bash 
python extract_fpfh.py 
python train.py 
```
