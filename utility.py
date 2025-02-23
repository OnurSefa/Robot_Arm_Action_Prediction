import torch
from torch.utils.data import Dataset
import environment
from homework1 import Hw1Env


class Loader:
    def __init__(self, data_dir, method, file_count=20):
        if method == "mlp":
            self.keywords = ["actions", "positions", 'imgs_after']
        elif method == "cnn":
            self.keywords = ['actions', 'imgs_before', 'positions', 'imgs_after']
        elif method == "raw":
            self.keywords = ['actions', 'imgs_before', 'imgs_after']
        self.data = {}
        for keyword in self.keywords:
            self.data[keyword] = []
        for i in range(file_count):
            for keyword in self.keywords:
                if 'imgs' in keyword:
                    self.data[keyword].append(torch.load(f"{data_dir}/{keyword}_{i}.pt").float() / 255)
                else:
                    self.data[keyword].append(torch.load(f"{data_dir}/{keyword}_{i}.pt"))
        for keyword in self.keywords:
            self.data[keyword] = torch.concat(self.data[keyword], dim=0).float()
        self.data_len = self.data[self.keywords[0]].shape[0]

    def take(self, train_ratio=0.8, eval_ratio=0.1):
        train_end = round(self.data_len * train_ratio)
        eval_end = round(self.data_len * (train_ratio + eval_ratio))
        limits = [[0, train_end], [train_end, eval_end], [eval_end, self.data_len]]
        output = []
        for limit in limits:
            current_output = {}
            for keyword in self.keywords:
                current_output[keyword] = self.data[keyword][limit[0]:limit[1]]
            output.append(current_output)
        return output


class RobotDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data['actions'].shape[0]

    def __getitem__(self, idx):
        output = []
        for key in self.data.keys():
            output.append(self.data[key][idx])
        return output


class VisualizeEnv(Hw1Env):
    def __init__(self, prediction, color, **kwargs):
        self.prediction = prediction
        if color == "red":
            self.color = [0.8, 0.2, 0.2, 1]
        elif color == "blue":
            self.color = [0.2, 0.2, 0.8, 1]
        else:
            self.color = [0.2, 0.2, 0.2, 1]
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        size = [0.025, 0.025, 0.025]
        environment.create_object(scene, "box", pos=[self.prediction[0], self.prediction[1], 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=self.color, friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def step(self, **kwargs):
        self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})

