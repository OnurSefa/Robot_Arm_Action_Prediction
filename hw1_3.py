import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utility import Loader, RobotDataset, VisualizeEnv
from torchvision.transforms.functional import to_pil_image
import torch.optim as o
import torch.nn.functional as F
import os
import shutil
import matplotlib.pyplot as plt
from pytorch_msssim import ssim


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        layers = [
            nn.Linear(128 * 16 * 16 + 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 16 * 16),
            nn.ReLU(inplace=True)
        ]
        self.linear = nn.Sequential(*layers)
        self.bottleneck = DoubleConv(128, 256)

    def forward(self, x, actions):
        x = torch.cat([x.reshape(x.shape[0], -1), actions], dim=1)
        x = self.linear(x).reshape(x.shape[0], 128, 16, 16)
        x = self.bottleneck(x)
        return x


class ModelRAW(nn.Module):
    def __init__(self):
        super(ModelRAW, self).__init__()
        self.enc1 = DoubleConv(7, 32, use_bn=False)
        self.res1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64, use_bn=False)
        self.res2 = ResidualBlock(64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128, use_bn=False)
        self.res3 = ResidualBlock(128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = Bottleneck()

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128, use_bn=False)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64, use_bn=False)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32, use_bn=False)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, imgs, actions):
        action_one_hot = F.one_hot(actions.long(), num_classes=4).float()
        actions_map = action_one_hot.view(-1, 4, 1, 1).expand(-1, 4, imgs.shape[2], imgs.shape[3])
        x = torch.cat([imgs, actions_map], dim=1)  # Now 7 channels

        enc1 = self.enc1(x)
        enc1 = self.res1(enc1)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        enc2 = self.res2(enc2)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        enc3 = self.res3(enc3)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3, action_one_hot)

        up3 = self.up3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)
        up2 = self.up2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)
        up1 = self.up1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(up1)
        out = self.out_conv(dec1)
        out = torch.sigmoid(out)
        return out


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, predicted, target):
        return 1 - ssim(predicted, target, data_range=self.data_range, size_average=True)


def train(epoch=30, eval_epoch_step=100, learning_rate=0.01, batch_size=256, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    loader = Loader(method='raw', data_dir="dataset", file_count=10)
    train_data, eval_data, _ = loader.take()
    train_dataset, eval_dataset = RobotDataset(train_data), RobotDataset(eval_data)
    train_loader, eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = ModelRAW().to(device)
    optimizer = o.Adam(model.parameters(), lr=learning_rate)
    # loss_function = PSNRLoss(device)
    loss_function = SSIMLoss()
    # loss_function = CompositeLoss()

    losses = {
        "train": {"x": [], "y": []},
        "eval": {"x": [], "y": []}
    }
    best_loss = np.inf
    best_path = None

    for e in range(epoch):
        batch_index = 0
        total_loss = 0
        total_count = 0
        model.train()
        for actions, imgs, post_imgs in train_loader:
            optimizer.zero_grad()

            actions = actions.to(device)
            imgs = imgs.to(device)
            post_imgs = post_imgs.to(device)

            predictions = model(imgs, actions)
            loss = loss_function(predictions, post_imgs)
            loss.backward()

            optimizer.step()
            print(f'Epoch {e}->\tBatch {batch_index}->\tLoss {loss.item():.6f}')
            total_loss += loss.item()
            total_count += 1
            batch_index += 1
        losses["train"]["x"].append(e)
        losses["train"]["y"].append(total_loss/total_count)

        output_path = f'{save_dir}/raw_{e}.pt'
        if e % eval_epoch_step == 0:
            torch.save(model.state_dict(), output_path)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_count = 0
            for actions, imgs, post_imgs in eval_loader:
                actions = actions.to(device)
                imgs = imgs.to(device)
                post_imgs = post_imgs.to(device)
                predictions = model(imgs, actions)
                loss = loss_function(predictions, post_imgs)
                total_loss += loss.item()
                total_count += 1
            eval_loss = total_loss/total_count
            print(f'Epoch {e}->\tEVAL LOSS {eval_loss:.6f}')
            losses["eval"]["x"].append(e)
            losses["eval"]["y"].append(eval_loss)
            if e % eval_epoch_step == 0:
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_path = output_path

    shutil.copy2(best_path, 'hw1_3.pt')
    print(f"best loss: {best_loss}\tbest path: {best_path}")
    for method in losses:
        plt.plot(losses[method]["x"], losses[method]["y"], label=f'{method} loss')
    plt.title(f"RAW Model Training Loss Graph")
    plt.legend()
    plt.savefig(f"raw_training.png", dpi=300)


def test(model_path='models/raw_199.pt', batch_size=256, example_count=5):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = ModelRAW()
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model = model.to(device)
    model.eval()

    loader = Loader(method='raw', data_dir="dataset", file_count=10)
    _, _, test_data = loader.take()
    test_dataset = RobotDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_function = SSIMLoss()
    # loss_function = CompositeLoss()

    actual_images = []
    predicted_images = []

    with torch.no_grad():
        total_loss = 0
        total_count = 0
        current_count = 0
        for actions, imgs, post_imgs in test_loader:
            actions = actions.to(device)
            imgs = imgs.to(device)
            post_imgs = post_imgs.to(device)
            predictions = model(imgs, actions)
            loss = loss_function(predictions, post_imgs)
            total_loss += loss.item()
            total_count += 1

            prediction_index = 0
            while current_count < example_count and prediction_index < predictions.shape[0]:
                predicted = (predictions[prediction_index].to('cpu') * 255).type(torch.uint8)
                predicted_img = to_pil_image(predicted, 'RGB')
                predicted_images.append(predicted_img)
                actual = (post_imgs[prediction_index].to('cpu') * 255).type(torch.uint8)
                actual_img = to_pil_image(actual, 'RGB')
                actual_images.append(actual_img)
                prediction_index += 1
                current_count += 1

        test_loss = total_loss/total_count
        print(f'Test Loss {test_loss:.6f}')

    fig, axes = plt.subplots(2, example_count, figsize=(2*example_count, 4))
    for i in range(example_count):
        axes[0, i].imshow(actual_images[i])
        axes[0, i].set_title(f"Real {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(predicted_images[i])
        axes[1, i].set_title(f"Predicted {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(f'raw_evaluation.png', dpi=300)


if __name__ == "__main__":
    train(epoch=2001, eval_epoch_step=100, learning_rate=0.00002)
    test('hw1_3.pt', example_count=10)
