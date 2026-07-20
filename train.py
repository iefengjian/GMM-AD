
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
import gc
import pandas as pd

from utils import UpsampleScores
from GMM_AD import GMM_AD_Wrapper
from dataset import FPFHData,shapenet3d_classes,mulsen_classes,real3d_classes

def run_cls(dataset_path,category, train_split,test_split, gmm_K, gmm_r,device):

    score_upsample = True
    density = GMM_AD_Wrapper(K=gmm_K, d_latent=gmm_r, iters=100, max_tokens=4096*200, device=device)

    train_dataset = FPFHData(class_name=category, split=train_split,
                               dataset_path=dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=4, drop_last=False, pin_memory=False)

    test_dataset = FPFHData(class_name=category, split=test_split,
                              dataset_path=dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, drop_last=False, pin_memory=True)

    # -------- collect (keep on CPU) --------
    for _, pack in tqdm(enumerate(train_loader), f"collect [{category}]"):
        fpfhs = pack["fpfhs"]                 # CPU
        density.collect(fpfhs)                # ensure collect doesn't store GPU tensors

    density.finalize()

    # -------- evaluate --------
    pixel_scores, pixel_labels = [], []
    obj_scores, obj_labels = [], []

    with torch.no_grad():
        for _, pack in tqdm(enumerate(test_loader), f"evaluate [{category}]"):
            points = pack["pc"].to(device, non_blocking=True)
            fpfhs  = pack["fpfhs"].to(device, non_blocking=True)
            centers = pack["centers"].to(device, non_blocking=True)

            mask  = pack["mask"]   # CPU
            label = pack["label"]  # CPU

            token_scores = density.score(fpfhs)
            if score_upsample:
                pixel_score, object_score = UpsampleScores(token_scores, points, centers)
            else:
                pixel_score=token_scores
                tmp_s,_ = torch.topk(token_scores, 80)
                object_score = torch.mean(tmp_s)
                 
            pixel_scores.append(pixel_score.detach().cpu().flatten())
            pixel_labels.append(mask.detach().cpu().flatten())

            obj_scores.append(object_score.detach().cpu().view(1))
            obj_labels.append(label.detach().cpu().view(1))

    pixel_scores = torch.cat(pixel_scores, dim=0)
    pixel_labels = torch.cat(pixel_labels, dim=0)
    obj_scores = torch.cat(obj_scores, dim=0)
    obj_labels = torch.cat(obj_labels, dim=0)


    pix_auc = roc_auc_score(pixel_labels.numpy(), pixel_scores.numpy())
    obj_auc = roc_auc_score(obj_labels.numpy(), obj_scores.numpy())

    del train_loader, test_loader, train_dataset, test_dataset
    del density
    torch.cuda.empty_cache()
    gc.collect()

    return float(obj_auc), float(pix_auc)


def main(ccfg, device):
    classes = ccfg["classes"]
    results = []
    for cls in classes:
        
        obj, pix = run_cls(dataset_path=ccfg["dataset_path"],
                           category=cls, 
                           train_split=ccfg["train_split"],
                           test_split=ccfg["test_split"], 
                           gmm_K=ccfg["K"],
                           gmm_r=ccfg["r"],
                           device=device,
                           )
        print(f"[{cls}] eval: obj_auc={obj:.6f}, pix_auc={pix:.6f}")
        results.append({
            "class": cls,
            "obj_auc": obj,
            "pix_auc": pix
        })
    df = pd.DataFrame(results)

    # compute mean
    mean_row = {
        "class": "mean",
        "obj_auc": df["obj_auc"].mean(),
        "pix_auc": df["pix_auc"].mean()
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # save to csv
    csv_path = ccfg["result_file"]
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    print(df)


if __name__=="__main__":

    cfg={
        "shape3d":{
            "K":7,
            "r":12,
            "classes":shapenet3d_classes(),
            "dataset_path":"datasets/Shape3D_fpfh",
            "train_split":"train",
            "test_split":"test",
            "result_file":"shape3d.csv"
        },

        "mulsen":{
            "K":7,
            "r":12,
            "classes":mulsen_classes(),
            "dataset_path":"datasets/Mulsen_fpfh",
            "train_split":"train",
            "test_split":"test",
            "result_file":"mulsen.csv"
        },

        "real3d":{
            "K":50,
            "r":16,
            "classes":real3d_classes(),
            "dataset_path":"datasets/Real3D_fpfh",
            "train_split":"train",
            "test_split":"test",
            "result_file":"real3d.csv"
        }
    }
    
    device="cuda:0"

    ccfg=cfg["shape3d"]
    main(ccfg,device)

    ccfg=cfg["mulsen"]
    main(ccfg,device)

    ccfg=cfg["real3d"]
    main(ccfg,device)
