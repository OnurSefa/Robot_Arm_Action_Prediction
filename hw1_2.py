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


class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 26, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.linear0 = nn.Linear(3146, 1024)
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 8)
        self.linear5 = nn.Linear(12, 2)

    def forward(self, imgs, actions):
        imgs = self.pool(self.relu(self.conv1(imgs)))
        imgs = self.pool(self.relu(self.conv2(imgs)))
        imgs = self.pool(self.relu(self.conv3(imgs)))
        imgs = imgs.flatten(start_dim=1)
        action_one_hot = F.one_hot(actions.long()).float()
        x = self.linear0(imgs)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = torch.concat([x, action_one_hot], dim=1)
        x = self.linear5(x)

        return x

def train(epoch=30, learning_rate=0.01, batch_size=256, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    loader = Loader(method='cnn', data_dir='dataset', file_count=10)
    train_data, eval_data, _ = loader.take()
    train_dataset, eval_dataset = RobotDataset(train_data), RobotDataset(eval_data)
    train_loader, eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = ModelCNN().to(device)
    optimizer = o.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

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
        for actions, imgs, positions, _ in train_loader:
            optimizer.zero_grad()

            actions = actions.to(device)
            imgs = imgs.to(device)
            positions = positions.to(device)

            predictions = model(imgs, actions)
            loss = loss_function(predictions, positions)
            loss.backward()

            optimizer.step()
            print(f'Epoch {e}->\tBatch {batch_index}->\tLoss {loss.item():.6f}')
            total_loss += loss.item()
            total_count += 1
            batch_index += 1
        losses["train"]["x"].append(e)
        losses["train"]["y"].append(total_loss/total_count)

        output_path = f'{save_dir}/cnn_{e}.pt'
        torch.save(model.state_dict(), output_path)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_count = 0
            for actions, imgs, positions, _ in eval_loader:
                actions = actions.to(device)
                imgs = imgs.to(device)
                positions = positions.to(device)
                predictions = model(imgs, actions)
                loss = loss_function(predictions, positions)
                total_loss += loss.item()
                total_count += 1
            eval_loss = total_loss/total_count
            print(f'Epoch {e}->\tEVAL LOSS {eval_loss:.6f}')
            losses["eval"]["x"].append(e)
            losses["eval"]["y"].append(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_path = output_path

    shutil.copy2(best_path, 'hw1_2.pt')
    print(f"best loss: {best_loss}\tbest path: {best_path}")
    for method in losses:
        plt.plot(losses[method]["x"], losses[method]["y"], label=f'{method} loss')
    plt.title(f"CNN Model Training Loss Graph")
    plt.legend()
    plt.savefig(f"cnn_training.png", dpi=300)


def test(model_path='cnn.pt', batch_size=256, example_count=5):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = ModelCNN()
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model = model.to(device)
    model.eval()

    loader = Loader(method='cnn', data_dir="dataset", file_count=10)
    _, _, test_data = loader.take()
    test_dataset = RobotDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.MSELoss()

    predicted_positions = []
    predicted_images = []
    actual_images = []

    with torch.no_grad():
        total_loss = 0
        total_count = 0
        current_count = 0
        for actions, imgs, positions, imgs_after in test_loader:
            actions = actions.to(device)
            imgs = imgs.to(device)
            positions = positions.to(device)
            predictions = model(imgs, actions)
            loss = loss_function(predictions, positions)
            total_loss += loss.item()
            total_count += 1

            prediction_index = 0
            while current_count < example_count and prediction_index < predictions.shape[0]:
                predicted_positions.append(predictions[prediction_index, :].to('cpu').numpy())
                actual = (imgs_after[prediction_index].to('cpu') * 255).type(torch.uint8)
                actual_img = to_pil_image(actual, 'RGB')
                actual_images.append(actual_img)
                prediction_index += 1
                current_count += 1

        test_loss = total_loss/total_count
        print(f'Test Loss {test_loss:.6f}')

    for position in predicted_positions:
        env = VisualizeEnv(position, color='blue', render_mode="offscreen")
        env.step()
        _, pixels = env.state()
        img = to_pil_image(pixels, 'RGB')
        predicted_images.append(img)
        del env

    fig, axes = plt.subplots(2, example_count, figsize=(2*example_count, 4))
    for i in range(example_count):
        axes[0, i].imshow(actual_images[i])
        axes[0, i].set_title(f"Real {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(predicted_images[i])
        axes[1, i].set_title(f"Predicted {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(f'cnn_evaluation.png', dpi=300)


if __name__ == "__main__":
    train(200, learning_rate=0.002)
    test('hw1_2.pt', example_count=10)
