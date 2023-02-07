from pathlib import Path
from torch_geometric.datasets import S3DIS

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

current_path = Path.cwd()
dataset_dir = current_path / "S3DIS"

train_dataset = S3DIS(dataset_dir, test_area=6, train=True, transform=None, pre_transform=None, pre_filter=None)
test_dataset = S3DIS(dataset_dir, test_area=6, train=False, transform=None, pre_transform=None, pre_filter=None)

from torch_geometric.nn import global_max_pool
import torch.nn as nn
from torch_geometric.data import DataLoader as DataLoader

class InputTNet(nn.Module):
    def __init__(self):
        super(InputTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 9)
        )
        
    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 3, 3)
        id_matrix = torch.eye(3).to(x.device).view(1, 3, 3).repeat(x.shape[0], 1, 1)
        x = id_matrix + x
        return x

class FeatureTNet(nn.Module):
    def __init__(self):
        super(FeatureTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        
    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 64, 64)
        id_matrix = torch.eye(64).to(x.device).view(1, 64, 64).repeat(x.shape[0], 1, 1)
        x = id_matrix + x
        return x


class PointNetSegmentation(nn.Module):
    def __init__(self):
        super(PointNetSegmentation, self).__init__()
        self.input_tnet = InputTNet()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.feature_tnet = FeatureTNet()
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1088, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 13)
        )
        
    def forward(self, batch_data):
        x = batch_data.pos
        
        input_transform = self.input_tnet(x, batch_data.batch)
        transform = input_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 3, 1)).view(-1, 3)
        
        x = self.mlp1(x)
        
        feature_transform = self.feature_tnet(x, batch_data.batch)
        transform = feature_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 64, 1)).view(-1, 64)
        pointwise_feature = x
        
        x = self.mlp2(x)        
        x = global_max_pool(x, batch_data.batch)
        global_feature = x[batch_data.batch, :]
        
        x = torch.cat([pointwise_feature, global_feature], axis=1)
        x = self.mlp3(x)
        
        return x, input_transform, feature_transform


import torch
from torch.utils.tensorboard import SummaryWriter

num_epoch = 400
batch_size = 16

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = PointNetSegmentation()
model = model.to(device)

optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epoch // 4, gamma=0.5)

log_dir = current_path / "log_S3DIS_segmentation"
log_dir.mkdir(exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criteria = torch.nn.CrossEntropyLoss()




from tqdm import tqdm

for epoch in range(num_epoch):
    model = model.train()
    
    losses = []
    for batch_data in tqdm(train_dataloader, total=len(train_dataloader)):
        batch_data = batch_data.to(device)
        this_batch_size = batch_data.batch.detach().max() + 1
        
        pred_y, _, feature_transform = model(batch_data)
        true_y = batch_data.y.detach()

        class_loss = criteria(pred_y, true_y)
        accuracy = float((pred_y.argmax(dim=1) == true_y).sum()) / float(this_batch_size)

        id_matrix = torch.eye(feature_transform.shape[1]).to(feature_transform.device).view(1, 64, 64).repeat(feature_transform.shape[0], 1, 1)
        transform_norm = torch.norm(torch.bmm(feature_transform, feature_transform.transpose(1, 2)) - id_matrix, dim=(1, 2))
        reg_loss = transform_norm.mean()

        loss = class_loss + reg_loss * 0.001
        
        losses.append({
            "loss": loss.item(),
            "class_loss": class_loss.item(),
            "reg_loss": reg_loss.item(),
            "accuracy": accuracy,
            "seen": float(this_batch_size)})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    if (epoch % 10 == 0):
        model_path = log_dir / f"model_{epoch:06}.pth"
        torch.save(model.state_dict(), model_path)
    
    loss = 0
    class_loss = 0
    reg_loss = 0
    accuracy = 0
    seen = 0
    for d in losses:
        seen = seen + d["seen"]
        loss = loss + d["loss"] * d["seen"]
        class_loss = class_loss + d["class_loss"] * d["seen"]
        reg_loss = reg_loss + d["reg_loss"] * d["seen"]
        accuracy = accuracy + d["accuracy"] * d["seen"]
    loss = loss / seen
    class_loss = class_loss / seen
    reg_loss = reg_loss / seen
    accuracy = accuracy / seen
    writer.add_scalar("train_epoch/loss", loss, epoch)
    writer.add_scalar("train_epoch/class_loss", class_loss, epoch)
    writer.add_scalar("train_epoch/reg_loss", reg_loss, epoch)
    writer.add_scalar("train_epoch/accuracy", accuracy, epoch)

    with torch.no_grad():
        model = model.eval()

        losses = []
        for batch_data in tqdm(test_dataloader, total=len(test_dataloader)):
            batch_data = batch_data.to(device)
            this_batch_size = batch_data.batch.detach().max() + 1

            pred_y, _, feature_transform = model(batch_data)
            true_y = batch_data.y.detach()

            class_loss = criteria(pred_y, true_y)
            accuracy =float((pred_y.argmax(dim=1) == true_y).sum()) / float(this_batch_size)

            id_matrix = torch.eye(feature_transform.shape[1]).to(feature_transform.device).view(1, 64, 64).repeat(feature_transform.shape[0], 1, 1)
            transform_norm = torch.norm(torch.bmm(feature_transform, feature_transform.transpose(1, 2)) - id_matrix, dim=(1, 2))
            reg_loss = transform_norm.mean()

            loss = class_loss + reg_loss * 0.001 * 0.001

            losses.append({
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                "reg_loss": reg_loss.item(),
                "accuracy": accuracy,
                "seen": float(this_batch_size)})
            
        loss = 0
        class_loss = 0
        reg_loss = 0
        accuracy = 0
        seen = 0
        for d in losses:
            seen = seen + d["seen"]
            loss = loss + d["loss"] * d["seen"]
            class_loss = class_loss + d["class_loss"] * d["seen"]
            reg_loss = reg_loss + d["reg_loss"] * d["seen"]
            accuracy = accuracy + d["accuracy"] * d["seen"]
        loss = loss / seen
        class_loss = class_loss / seen
        reg_loss = reg_loss / seen
        accuracy = accuracy / seen
        writer.add_scalar("test_epoch/loss", loss, epoch)
        writer.add_scalar("test_epoch/class_loss", class_loss, epoch)
        writer.add_scalar("test_epoch/reg_loss", reg_loss, epoch)
        writer.add_scalar("test_epoch/accuracy", accuracy, epoch)