#!/usr/bin/env python

import gc
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from decord import VideoReader, cpu
from scipy.stats import spearmanr
from peft import get_peft_model, LoraConfig, TaskType
from src.mri import Benchmark
from src import video_ops, tools
from deepjuice.extraction import FeatureExtractor
from deepjuice.systemops.devices import cuda_device_report


class X3DTripletModel(nn.Module):
    def __init__(self, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
        self.backbone.blocks[-1] = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 13, 160, 160)
            features = self.backbone(dummy)
            pooled = self.global_pool(features)
            self.output_dim = pooled.view(1, -1).shape[1]

        self.embedding_layer = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self._wrap_with_lora(lora_r, lora_alpha, lora_dropout)
        self._log_param_counts()

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flat = pooled.view(pooled.size(0), -1)
        return self.embedding_layer(flat)

    def _wrap_with_lora(self, r, alpha, dropout):
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["conv", "proj", "fc", "linear"]
        )
        self.backbone = get_peft_model(self.backbone, config)

    def _log_param_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"🧠 Total parameters:     {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        print(f"❄️  Frozen parameters:   {frozen_params:,}")


class TripletDataset(Dataset):
    def __init__(self, triplet_df, video_dir, num_frames=16, image_size=(224, 224)):
        self.triplets = triplet_df[['stim1_name', 'stim2_name', 'stim3_name', 'choice']].values
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def get_vid_names(self, stim1, stim2, stim3, choice):
        vid_map = pd.read_csv('/home/kgarci18/scratch4-lisik3/kgarci18/SIfMRI_modeling/data/interim/similarity/video_mapping.csv')
        stim1 = vid_map.loc[stim1, 'video_name'][:-4]
        stim2 = vid_map.loc[stim2, 'video_name'][:-4]
        stim3 = vid_map.loc[stim3, 'video_name'][:-4]
        choice = vid_map.loc[choice, 'video_name'][:-4]
        return stim1, stim2, stim3, choice

    def sample_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = []
        for i in indices:
            frame = vr[i].cpu().numpy()
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        return frames.permute(1, 0, 2, 3)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        stim1, stim2, stim3, choice = self.triplets[idx]
        stim1, stim2, stim3, choice = self.get_vid_names(stim1, stim2, stim3, choice)
        stimuli_set = {stim1, stim2, stim3}
        negative = choice
        similar_pair = list(stimuli_set - {negative})
        if len(similar_pair) != 2:
            raise ValueError(f"Invalid triplet assignment at index {idx}: {stim1, stim2, stim3, choice}")
        anchor, positive = similar_pair
        return self.sample_frames(f"{self.video_dir}/{anchor}.mp4"), \
               self.sample_frames(f"{self.video_dir}/{positive}.mp4"), \
               self.sample_frames(f"{self.video_dir}/{negative}.mp4")


class VideoSimilarityFinetuning:
    def __init__(self, args):
        self.process = 'VideoSimilarityFinetuning'
        self.overwrite = args.overwrite
        self.model_name = args.model_name
        self.data_dir = args.data_dir
        self.user = args.user
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.pt'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def rsa_loss(self, model_rsm, human_rsm_flat):
        model_rsm_flat = model_rsm[torch.triu_indices(model_rsm.shape[0], model_rsm.shape[1], offset=1).unbind()]
        model_rsm_flat = (model_rsm_flat - model_rsm_flat.mean()) / model_rsm_flat.std()
        human_rsm_flat = (human_rsm_flat - human_rsm_flat.mean()) / human_rsm_flat.std()
        return -torch.sum(model_rsm_flat * human_rsm_flat) / (len(human_rsm_flat) - 1)

    def compute_model_rsm(self, model, dataloader, grad=False):
        embs = []
        if not grad:
            model.eval()
            with torch.no_grad():
                for a, _, _ in dataloader:
                    a = a.to(self.device)
                    embs.append(model(a).cpu())
        else:
            model.train()
            for a, _, _ in dataloader:
                a = a.to(self.device)
                embs.append(model(a))
        all_embs = torch.cat(embs, dim=0).to(self.device)
        all_embs = all_embs - all_embs.mean(dim=1, keepdim=True)
        normed = all_embs / (all_embs.std(dim=1, keepdim=True) + 1e-8)
        corr_matrix = torch.matmul(normed, normed.T) / normed.shape[1]
        return 1 - corr_matrix

    def collate_fn(self, batch):
        anchors, positives, negatives = zip(*batch)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def run(self):
        tools.send_slack(f'Started training {self.model_name}', channel=self.user)
        sim_judg = pd.read_csv(f'{self.data_dir}/interim/similarity/train_triplets.csv')
        sim_judge_train_rsm = pd.read_csv(f'{self.data_dir}/interim/similarity/sim_judge_train_rsm.csv', index_col=0)
        train_ids, val_ids = train_test_split(sim_judge_train_rsm.index, test_size=0.2, random_state=42)

        train_triplets = sim_judg[
            sim_judg[['stim1_name', 'stim2_name', 'stim3_name', 'choice']].apply(lambda row: all(x in train_ids for x in row), axis=1)]
        val_triplets = sim_judg[
            sim_judg[['stim1_name', 'stim2_name', 'stim3_name', 'choice']].apply(lambda row: all(x in val_ids for x in row), axis=1)]

        train_rsm = sim_judge_train_rsm.loc[train_ids, train_ids]
        val_rsm = sim_judge_train_rsm.loc[val_ids, val_ids]

        def flatten_rsm(rsm):
            tri = np.triu_indices_from(rsm, k=1)
            return torch.tensor(rsm.values[tri], dtype=torch.float32).to(self.device)

        sim_train_flat = flatten_rsm(train_rsm)
        sim_val_flat = flatten_rsm(val_rsm)

        train_dataset = TripletDataset(train_triplets, f'{self.data_dir}/raw/videos')
        val_dataset = TripletDataset(val_triplets, f'{self.data_dir}/raw/videos')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=self.collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)

        model = X3DTripletModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        scaler = GradScaler()
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

        best_val = -float('inf')
        early_stop = 0

        for epoch in range(50):
            print(f"Epoch {epoch}")
            model.train()
            running_loss = 0

            model_rsm_train = self.compute_model_rsm(model, train_loader, grad=True)
            loss_rsa_epoch = self.rsa_loss(model_rsm_train, sim_train_flat)

            for anchors, positives, negatives in tqdm(train_loader):
                anchors, positives, negatives = anchors.to(self.device), positives.to(self.device), negatives.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    ea, ep, en = model(anchors), model(positives), model(negatives)
                    loss_triplet = triplet_loss_fn(ea, ep, en)
                    loss = 0.7 * loss_triplet + 0.3 * loss_rsa_epoch
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_rsm = self.compute_model_rsm(model, val_loader, grad=False)
                val_rsm_flat = val_rsm[torch.triu_indices(val_rsm.shape[0], val_rsm.shape[1], offset=1).unbind()].cpu().numpy()
                val_score, _ = spearmanr(sim_val_flat.cpu().numpy(), val_rsm_flat)

            print(f"Validation RSA: {val_score:.4f}")
            scheduler.step(val_score)

            if val_score > best_val:
                best_val = val_score
                early_stop = 0
                torch.save(model.state_dict(), f"{self.out_file}_best.pt")
            else:
                early_stop += 1
                if early_stop >= 5:
                    break

            torch.cuda.empty_cache()
            gc.collect()

        tools.send_slack(f"Finished {self.model_name} - Best RSA: {best_val:.4f}", channel=self.user)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, default='kgarci18')
    parser.add_argument('--model_name', type=str, default='LoRA_X3D')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    VideoSimilarityFinetuning(args).run()


if __name__ == "__main__":
    main()
