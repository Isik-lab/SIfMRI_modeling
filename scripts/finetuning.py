# /Applications/anaconda3/envs/deepjuice/bin/python

import gc
import torch
import time
import argparse
import pandas as pd
import numpy as np
import os
from src.mri import Benchmark
from src import video_ops, tools
from deepjuice.extraction import FeatureExtractor
from pathlib import Path
from deepjuice.systemops.devices import cuda_device_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from decord import VideoReader, cpu
from scipy.stats import spearmanr
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class X3DTripletModel(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        flat = pooled.view(pooled.size(0), -1)
        return self.embedding_layer(flat)

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
        vid_map = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/code/similarity-judgements/raw/video_mapping.csv')
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
        frames = torch.stack(frames)  # (T, C, H, W)
        return frames.permute(1, 0, 2, 3)  # ➡️ (C, T, H, W)

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
        anchor_video = f"{self.video_dir}/{anchor}.mp4"
        pos_video = f"{self.video_dir}/{positive}.mp4"
        neg_video = f"{self.video_dir}/{negative}.mp4"
        anchor_frames = self.sample_frames(anchor_video)
        positive_frames = self.sample_frames(pos_video)
        negative_frames = self.sample_frames(neg_video)
        return anchor_frames, positive_frames, negative_frames

class VideoSimilarityFinetuning:
    def __init__(self, args):
        self.process = 'VideoSimilarityFinetuning'
        self.overwrite = args.overwrite
        self.model_name = args.model_name
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        self.user = args.user
        self.memory_limit = args.memory_limit
        self.extension = 'mp4'
        print(vars(self))
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.parquet'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')

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
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        return anchors, positives, negatives

    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name} on Rockfish...', channel=self.user)
                # Load data and sort
                sim_judg = pd.read_csv(f'{self.data_dir}/interim/similarity/train_triplets.csv')
                sim_judge_train_rsm = pd.read_csv(f'{self.data_dir}/interim/similarity/sim_judge_train_rsm.csv')
                train_idx = pd.read_csv(f'{self.data_dir}/interim/similarity/train_idx.csv')
                print('Loaded files!')

                triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

                # ===========================
                # ⚙️ Training Setup
                # ===========================
                print('Loading Model...')
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = X3DTripletModel().to(self.device)
                print('Loaded Model!')

                print('Setting up training variables...')
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
                scaler = GradScaler()

                num_epochs = 50
                freeze_backbone = True
                unfreeze_at = 10
                patience = 5

                best_val_score = -float('inf')
                early_stop_counter = 0
                best_epoch = 0

                train_losses = []
                val_scores = []
                lrs = []

                # 🚀 Load Dataloader
                print('Loading triplet loader...')
                video_dir = '/content/drive/MyDrive/Colab_Notebooks/code/SIfMRI_modeling/data/raw/videos'
                triplet_dataset = TripletDataset(sim_judg, video_dir)
                train_loader = DataLoader(triplet_dataset, batch_size=2, shuffle=True, collate_fn=self.collate_fn)
                print('Loaded triplet loader!')

                # 1. Split RSM indices
                all_indices = sim_judge_train_rsm.index.to_list()
                train_ids, val_ids = train_test_split(all_indices, test_size=0.2, random_state=42)

                # 2. Split triplets based on set membership
                def filter_triplets(df, id_set):
                    return df[
                        df['stim1_name'].isin(id_set) &
                        df['stim2_name'].isin(id_set) &
                        df['stim3_name'].isin(id_set) &
                        df['choice'].isin(id_set)
                        ].copy()

                train_triplets = filter_triplets(sim_judg, train_ids)
                val_triplets = filter_triplets(sim_judg, val_ids)

                # 3. Create train and val RSMs
                train_rsm = sim_judge_train_rsm.loc[train_ids, train_ids]
                val_rsm = sim_judge_train_rsm.loc[val_ids, val_ids]

                # 4. Flatten upper triangles
                def flatten_rsm(rsm):
                    tri = np.triu_indices_from(rsm, k=1)
                    flat = torch.tensor(rsm.values[tri], dtype=torch.float32)
                    return flat.to(device)

                sim_train_flat = flatten_rsm(train_rsm)
                sim_val_flat = flatten_rsm(val_rsm)

                # 5. Create DataLoaders
                train_dataset = TripletDataset(train_triplets, video_dir)
                val_dataset = TripletDataset(val_triplets, video_dir)
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

                # ===========================
                # 🚀 Training Loop
                # ===========================
                alpha = 0.7
                beta = 0.3

                print('Starting training...')
                for epoch in range(num_epochs):
                    print(f'Finetuning Epoch: {epoch}')
                    model.train()
                    running_loss = 0.0

                    if freeze_backbone and epoch == unfreeze_at:
                        for param in model.backbone.parameters():
                            param.requires_grad = True
                        freeze_backbone = False

                    # ⚡ Compute model RSM once per epoch (in training mode to allow gradient tracking if needed)
                    model_rsm_train = self.compute_model_rsm(model, train_loader, grad=True)
                    loss_rsa_epoch = self.rsa_loss(model_rsm_train, sim_train_flat)

                    for anchors, positives, negatives in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                        anchors, positives, negatives = anchors.to(self.device), positives.to(self.device), negatives.to(self.device)

                        optimizer.zero_grad()
                        with autocast():
                            ea, ep, en = model(anchors), model(positives), model(negatives)
                            loss_triplet = triplet_loss_fn(ea, ep, en)
                            total_loss = alpha * loss_triplet + beta * loss_rsa_epoch

                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        train_losses.append(total_loss.item())
                        running_loss += total_loss.item()

                    # 💡
                    print('Running Validation set...')
                    with torch.no_grad():
                        model.eval()
                        val_rsm = self.compute_model_rsm(model, val_loader, grad=False)
                        val_rsm_flat = val_rsm[
                            torch.triu_indices(val_rsm.shape[0], val_rsm.shape[1], offset=1).unbind()].cpu().numpy()
                        sim_val = sim_val_flat.cpu().numpy()
                        val_corr, _ = spearmanr(sim_val, val_rsm_flat)
                        val_scores.append(val_corr)
                        lrs.append(optimizer.param_groups[0]['lr'])

                        scheduler.step(val_corr)

                        # 💾 Save model if best
                        if val_corr > best_val_score:
                            early_stop_counter = 0
                            best_val_score = val_corr
                            best_epoch = epoch
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': running_loss / len(train_loader),
                                'val_rsa': val_corr
                            }, f"{self.out_file}_best_checkpoint_hybrid.pt")
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                break

                        print('Emptying cache...')
                        torch.cuda.empty_cache()
                        gc.collect()
                        print('Loop Complete!')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: {self.process} {self.model_name} in {elapsed} :baby-yoda:', channel=self.user)
                print(f"✅ Hybrid model best val RSA: {best_val_score:.4f} at epoch {best_epoch}")
                tools.send_slack(f"✅ Hybrid model best val RSA: {best_val_score:.4f} at epoch {best_epoch}",
                                 channel=self.user)
        except Exception as err:
            print(err)
            tools.send_slack(f'Error: {self.process} {self.model_name} Error = {err}', channel=self.user)
            raise err

def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='kgarci18')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args

    parser.add_argument('--model_name', type=str, default='No_Model')
    parser.add_argument('--model_input', type=str, default='videos')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling/data')
    parser.add_argument('--memory_limit', type=str, default=None)
    args = parser.parse_args(remaining_argv)
    VideoSimilarityFinetuning(args).run()


if __name__ == '__main__':
    main()
