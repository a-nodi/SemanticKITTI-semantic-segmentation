# -*- coding: utf-8 -*-
"""_summary_

Returns:
    _type_: _description_
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import yaml
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from kitti_dataloader import SemenaticKITTIDataset
from torchmetrics.classification import MulticlassJaccardIndex
import MinkowskiEngine as ME
from minkunet import MinkUNet34C
# from sklearn.metrics import jaccard_score, confusion_matrix

SEMANTIC_KITTI_PATH = "semantic_KITTI/dataset/"


def read_yaml(path):
    """
    
    """
    content = []
    with open(path, "r") as stream:
        content = yaml.safe_load(stream)

    return content


def collate_fn(batch) -> dict:
    """
    
    """
    coords, feats, labels = list(zip(*batch))

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch


def make_checkpoint(epoch, net_state_dict, optimizer_state_dict) -> dict:
    checkpoint = {
                    "model": "Unet",
                    "epoch": epoch + 1,
                    "model_state_dict": net_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                }

    return checkpoint


def train(config, start_epoch, train_dataloader, net, optimizer, criterion, device):
    now = dt.now().strftime('%Y%m%d%H%M%S')                     
    net.train()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(start_epoch, config.max_epochs)):
        pbar = tqdm(train_dataloader)

        for i, data in enumerate(pbar):
            coords, feats, labels = data # TODO: -100 label 다 지우고 돌려보기
            
            labels = labels.to(device)
            out = net(ME.SparseTensor(feats, coords, device=device))
            optimizer.zero_grad()
            
            loss = criterion(out.F, labels.long())
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "loss": loss.item()
                })
            wandb.log({
                "loss": loss.item()
            })

            del coords, feats, labels, out, loss
            torch.cuda.empty_cache()
            
        if config.save_checkpoint:    
            torch.save(
                make_checkpoint(epoch, net.state_dict(), optimizer.state_dict()),
                os.path.join("scripts", "warmup", "checkpoints", f"{now}_checkpoint_{epoch}.pt")
                )
        

def eval(config, val_dataloader, net, criterion, device):

    # net.eval()
    miou_array = np.array([])
    jaccard = MulticlassJaccardIndex(num_classes=19, average="micro", ignore_index=-100).to(device)
    torch.no_grad()
    pbar = tqdm(val_dataloader)
    
    for i, data in enumerate(pbar):
        coords, feats, labels = data

        labels = labels.to(device)
        out = net(ME.SparseTensor(feats, coords, device=device))
        # loss = criterion(out.F, labels.long()).item()
        pred = torch.argmax(torch.transpose(out.F, 0, 1), dim=0)
        miou = jaccard(pred, labels.long()).item()  # TODO: Bincount Error
        np.append(miou_array, [miou])

        pbar.set_postfix({
            "mIoU": miou
            })
        wandb.log({
            "mIoU": miou
            })
        
    wandb.log({
        "min mIoU": np.min(miou_array),
        "max mIoU": np.max(miou_array),
        "average mIoU": np.mean(miou_array)
        })


def main(config):
    """
    1. [SOLVED] yaml 읽어서 label 및 train, val 구분
    2. [SOLVED] Unet 생성 (6 짰으면) weight load, 아니면 새로 생성
    3. [SOLVED] train, val dataset 선언
    3.1. [SOLVED] collate_fn
    3.2. [SOLVED] noise 제거
    4. [SOLVED] dataset load 선언
    4.1. [SOLVED] voxel size divid
    5. [SOLVED] Adam optim 선언
    6. [SOLVED] (container 터질수도 있으니) 일정 epoch마다 weight 임시저장
    7. [SOLVED] sequence 마다 epoch, scene 8(4, 2)개 묶어서 minibatch
    8. [SOLVED] sparse tensor로 data load하고 feed forwawrd 
    8.1 [WIP] DDP
    9. [SOLVED] CE loss 계산 (scene마다 label과 어케 연산할지 생각해야함)
    10. [SOLVED] backprop
    11. [SOLVED] val에서 scene마다 mIoU 계산
    12. [SOLVED] WandB 연결
        
    """

    now = dt.now().strftime('%Y%m%d%H%M%S')
    
    wandb.init(
        project=config.wandb_project_name,
        notes=f"date: {now}"
    )
    wandb.config.update(config)

    device = torch.device('cuda:0')
    content = read_yaml(os.path.join(SEMANTIC_KITTI_PATH, "semantic-kitti.yaml"))
    
    net = MinkUNet34C(
        3,  # In nchannel
        19,  # Out nchannel (1 ~ 19)
        D=3  # Dimention
        ).to(device)
      
    optimizer = optim.Adam(
        params=net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
        )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)

    # Load checkpoint if exists
    if config.load_checkpoint:
        checkpoint = torch.load(os.path.join("scripts", "warmup", "log", config.checkpoint_name))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    
    else:
        start_epoch = 0

    # train if necessary
    if config.is_train:
        train_dataset = SemenaticKITTIDataset(
            SEMANTIC_KITTI_PATH,
            content['split']['train'],
            'train',
            content['learning_map'],
            voxel_size=config.voxel_size
            )
        
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=config.num_workers
            )
        
        train(config, start_epoch, train_dataloader, net, optimizer, criterion, device)

    # evaluate mIoU
    val_dataset = SemenaticKITTIDataset(
        SEMANTIC_KITTI_PATH, 
        content['split']['valid'], 
        'val', 
        content['learning_map'], 
        voxel_size=config.voxel_size
        )
        
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=config.num_workers
        )    
    
    eval(config, val_dataloader, net, criterion, device)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default=SEMANTIC_KITTI_PATH, type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int)  # Batch size
    parser.add_argument('--max_epochs', dest='max_epochs', default=20, type=int)  # Max epochs
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)  # Learning rate
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)  # Optimizer weight decay
    parser.add_argument('--voxel_size', dest='voxel_size', type=float, default=0.05)
    parser.add_argument('--is_train', dest='is_train', type=bool, default=True)
    parser.add_argument('--save_checkpoint', dest='save_checkpoint', type=bool, default=True)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=24)
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', type=str, default="20230730080852_checkpoint_6.pt")
    parser.add_argument('--wandb_project_name', dest='wandb_project_name', type=str, default="SemanticKITTI semantic segmentation")
    
    main(parser.parse_args())
