import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
from encode import rle_encode, list_to_string
import random

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch + out_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 8)
        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)
        self.up4 = Up(base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

def dice_score(y_p, y_t, smooth=1e-6):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_p = F.softmax(y_p, dim=1)
    y_p = torch.argmax(y_p, dim=1, keepdim=True)
    i = torch.sum(y_p * y_t, dim=(2, 3))
    u = torch.sum(y_p, dim=(2, 3)) + torch.sum(y_t, dim=(2, 3))
    score = (2 * i + smooth)/(u + smooth)
    return torch.mean(score)

def ce_loss(y_p, y_t):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_t = y_t.squeeze(dim=1)
    weight = torch.Tensor([0.1, 15.0]).to(y_t.device)
    criterion = nn.CrossEntropyLoss(weight)
    loss = criterion(y_p, y_t)
    return loss

def false_color(band11, band14, band15):
    def normalize(band, bounds):
        return (band - bounds[0]) / (bounds[1] - bounds[0])    
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)
    r = normalize(band15 - band14, _TDIFF_BOUNDS)
    g = normalize(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize(band14, _T11_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=2), 0, 1)

class ICRGWDataset(Dataset):
    def __init__(self, tar_path, ids, padding_size, augment=False):
        self.tar_path = tar_path
        self.ids = ids
        self.padding_size = padding_size
        self.augment = augment
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        N_TIMES_BEFORE = 4
        sample_path = f"{self.tar_path}/{self.ids[idx]}"
        band11 = np.load(f"{sample_path}/band_11.npy")[..., N_TIMES_BEFORE]
        band14 = np.load(f"{sample_path}/band_14.npy")[..., N_TIMES_BEFORE]
        band15 = np.load(f"{sample_path}/band_15.npy")[..., N_TIMES_BEFORE]
        image = false_color(band11, band14, band15)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        padding_size = self.padding_size
        image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
        try:
            label = np.load(f"{sample_path}/human_pixel_masks.npy")
            label = torch.Tensor(label).to(torch.int64)
            label = label.permute(2, 0, 1)
        except FileNotFoundError:
            label = torch.zeros((1, image.shape[1], image.shape[2]))
        if self.augment:
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[2])
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])
                label = torch.flip(label, dims=[1])
        return image, label


if __name__ == "__main__":

    data_path = "./train" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_ids = os.listdir(data_path)
        
    ids_train, ids_valid = train_test_split(image_ids, test_size=0.1, random_state=42)
    print(f"TrainSize: {len(ids_train)}, ValidSize: {len(ids_valid)}")

    batch_size = 2
    epochs = 50
    lr = 1e-4
    
    train_dataset = ICRGWDataset(data_path, ids_train, 2, augment=True)
    valid_dataset = ICRGWDataset(data_path, ids_valid, 2)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=None, num_workers=1)

    model = UNet(n_channels=3, n_classes=2, base_ch=32)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    bst_dice = 0
    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_dataloader)
        tot_loss = 0
        tot_score = 0
        count = 0
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = ce_loss(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss += loss.item()
            tot_score += dice_score(pred, y)
            count += 1
            bar.set_postfix(TrainLoss=f'{tot_loss/count:.4f}', TrainDice=f'{tot_score/count:.4f}')

        model.eval()
        bar = tqdm(valid_dataloader)
        tot_score = 0
        count = 0
        with torch.no_grad():
            for X, y in bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                tot_score += dice_score(pred, y)
                count += 1
                bar.set_postfix(ValidDice=f'{tot_score/count:.4f}')

        if tot_score/count > bst_dice:
            bst_dice = tot_score/count
            torch.save(model.state_dict(), 'u-net.pth')
            print(f"Epoch {epoch+1}: new best model saved! Valid Dice: {bst_dice:.4f}")

    # evaluate model on validation set
    model.load_state_dict(torch.load('u-net.pth'))
    model.eval()
    tot_score = 0
    with torch.no_grad():
        for X, y in valid_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            tot_score += dice_score(pred, y)
    print(f"Validation Dice Score: {tot_score/len(valid_dataloader):.4f}")

    submission = pd.read_csv('sample_submission.csv', index_col='record_id')
    test_dataset = ICRGWDataset("test/", os.listdir('test'), 2)
    model.eval()
    with torch.no_grad():
        for idx, (X, y) in enumerate(test_dataset):
            X = X.to(device)
            pred = model(X.unsqueeze(0))[:, :, 2:-2, 2:-2]
            pred = torch.argmax(pred, dim=1)[0]
            pred = pred.detach().cpu().numpy()
            submission.loc[int(test_dataset.ids[idx]), 'encoded_pixels'] = list_to_string(rle_encode(pred))
    submission.to_csv('submission.csv')
