import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from utils import get_score, matplot_auc, matplot_recall, matplot_prec, matplot_acc, matplot_loss, matplot_F1
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True

# 多任务数据加载
class MultiTaskDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        # Define the Transforms
        self.transform = transform
        # Set Inputs and Labels
        self.image_paths = os.listdir(image_paths)
        self.images = []
        # 纯合子（gt11） hom：false：0，true：1，ins：0，del：1
        # 杂合子（gt01） het：false：2，true：3，ins：2，del：3
        self.isPositive = []
        self.type = []
        self.filename = []

        for path in self.image_paths:
            filename = path.split("\\")[-1].split('.')[0] # chr1_24370-25116_gt01_del_true
            self.filename.append(filename)
            messages = filename.split("_")
            if len(messages) > 4 and messages[2] != '' and messages[3] != '' and messages[4] != '':
                if 'true' == messages[4]:continue
                if 'gt11' == messages[2] : # 纯和
                    self.images.append(os.path.join(image_paths, path))
                    if 'true' == messages[4]:
                        self.isPositive.append(1)
                    else:
                        self.isPositive.append(0)
                    if 'ins' == messages[3]:
                        self.type.append(0)
                    else:
                        self.type.append(1)
                elif 'gt01' == messages[2]: # 杂合
                    self.images.append(os.path.join(image_paths, path))
                    if 'true' == messages[4]:
                        self.isPositive.append(1+2)
                    else:
                        self.isPositive.append(0+2)
                    if 'ins' == messages[3]:
                        self.type.append(0+2)
                    else:
                        self.type.append(1+2)

    def __getitem__(self, index):
        # Load an Image
        img = Image.open(self.images[index]).convert('RGB')
        # Transform it
        img = self.transform(img)
        sample = {
            'image':img,'isPositive': self.isPositive[index], 'type': self.type[index], 'filename':self.filename[index]
        }
        return sample

    def __len__(self):
        return len(self.images)


class FourParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        # self.net.avgpool = nn.Identity()
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity() # 占位作用，没有具体含义

        self.shared_fc12 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU()
        )
        self.shared_fc34 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU()
        )

        self.net.fc1 = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//2, self.n_features//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//4, self.n_features//8, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(self.n_features//8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.net.fc2 = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//2, self.n_features//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//4, self.n_features//8, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(self.n_features//8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.net.fc3 = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//2, self.n_features//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//4, self.n_features//8, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(self.n_features//8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.net.fc4 = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//2, self.n_features//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.n_features//4, self.n_features//8, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(self.n_features//8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )


    def forward(self, x):
        x = self.net(x)

        homIsPositive_head = self.net.fc1(x.unsqueeze(-1).unsqueeze(-1))
        homType_head = self.net.fc2(x.unsqueeze(-1).unsqueeze(-1))

        hetIsPositive_head = self.net.fc3(x.unsqueeze(-1).unsqueeze(-1))
        hetType_head = self.net.fc4(x.unsqueeze(-1).unsqueeze(-1))
        return homIsPositive_head, homType_head, hetIsPositive_head, hetType_head


class AttentionParallelNet(nn.Module):
    def __init__(self, num_tasks):
        dropout_rate=0.1
        super(AttentionParallelNet, self).__init__()
        self.num_tasks = num_tasks
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()

        # Shared attention mechanism
        self.attention_weights = nn.Parameter(torch.ones(num_tasks))

        # Task-specific networks with Dropout
        self.task_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_features, self.n_features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),  # Add dropout here
                nn.Conv2d(self.n_features // 2, self.n_features // 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),  # Add dropout here
                nn.Conv2d(self.n_features // 4, self.n_features // 8, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(self.n_features // 8, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),  # Add dropout here
                nn.Linear(32, 2)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        x = self.net(x)
        attention_scores = F.softmax(self.attention_weights, dim=0)

        task_outputs = []
        for i, task_net in enumerate(self.task_networks):
            task_output = task_net(x.unsqueeze(-1).unsqueeze(-1))
            task_outputs.append(attention_scores[i] * task_output)

        return task_outputs
