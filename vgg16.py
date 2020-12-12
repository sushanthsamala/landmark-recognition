import gc
gc.enable()
import os
import sys
import math
import json
import time
import random
from glob import glob
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision
import torchvision.models as models
from torch import Tensor
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm



import sklearn

import warnings
warnings.filterwarnings("ignore")

import csv
import pprint
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from sklearn.model_selection import train_test_split

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MIN_SAMPLES_PER_CLASS = 150
BATCH_SIZE = 32
NUM_WORKERS = 20
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 1
LOG_FREQ = 10
NUM_TOP_PREDICTS = 5


train = pd.read_csv('/home/jingshuai/桌面/input/landmark-recognition-2020/train.csv')
test = pd.read_csv('/home/jingshuai/桌面/input/landmark-recognition-2020/sample_submission.csv')
train_dir = '/home/jingshuai/桌面/input/landmark-recognition-2020/train/'
test_dir = '/home/jingshuai/桌面/input/landmark-recognition-2020/test/'


# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, f, filters, s):
#         super(ConvBlock, self).__init__()
#         F1, F2, F3 = filters
#         self.stage = nn.Sequential(
#             nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
#             nn.BatchNorm2d(F1),
#             nn.ReLU(True),
#             nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ReLU(True),
#             nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(F3),
#         )
#         self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
#         self.batch_1 = nn.BatchNorm2d(F3)
#         self.relu_1 = nn.ReLU(True)
#
#     def forward(self, X):
#         X_shortcut = self.shortcut_1(X)
#         X_shortcut = self.batch_1(X_shortcut)
#         X = self.stage(X)
#         X = X + X_shortcut
#         X = self.relu_1(X)
#         return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result
# class ResModel(nn.Module):
#     def __init__(self, n_class):
#         super(ResModel, self).__init__()
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(3, 2, padding=1),
#         )
#         self.stage2 = nn.Sequential(
#             ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
#             IndentityBlock(256, 3, [64, 64, 256]),
#             IndentityBlock(256, 3, [64, 64, 256]),
#         )
#         self.stage3 = nn.Sequential(
#             ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
#             IndentityBlock(512, 3, [128, 128, 512]),
#             IndentityBlock(512, 3, [128, 128, 512]),
#             IndentityBlock(512, 3, [128, 128, 512]),
#         )
#         self.stage4 = nn.Sequential(
#             ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#         )
#         self.stage5 = nn.Sequential(
#             ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
#             IndentityBlock(2048, 3, [512, 512, 2048]),
#             IndentityBlock(2048, 3, [512, 512, 2048]),
#         )
#         self.pool = nn.AvgPool2d(2, 2, padding=1)
#         self.fc = nn.Sequential(
#             nn.Linear(8192, n_class)
#         )
#
#     def forward(self, X):
#         out = self.stage1(X)
#         out = self.stage2(out)
#         out = self.stage3(out)
#         out = self.stage4(out)
#         out = self.stage5(out)
#         out = self.pool(out)
#         out = out.view(out.size(0), 8192)
#         out = self.fc(out)
#         return out

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, mode: str):
        self.df = dataframe
        self.mode = mode
        self.image_dir = image_dir

        transforms_list = []
        if self.mode == 'train':
            # Increase image size from (64,64) to higher resolution,
            # Make sure to change in RandomResizedCrop as well.
            transforms_list.append(
                transforms.Resize(256),
                )
            transforms_list.append(
                transforms.CenterCrop(224),
            )
            transforms_list.append(
                transforms.ToTensor(),
            )
            transforms_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        else:
            transforms_list.extend([
                # Keep this resize same as train
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int):
        image_id = self.df.iloc[index].id
        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        image = Image.open(image_path)
        image = self.transforms(image)

        if self.mode == 'test':
            return {'image': image}
        else:
            return {'image': image,
                    'target': self.df.iloc[index].landmark_id}

    def __len__(self) -> int:
        return self.df.shape[0]


def load_data(train, test, train_dir, test_dir):
    counts = train.landmark_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)

    train = train.loc[train.landmark_id.isin(selected_classes)]
    print('train_df', train.shape)
    print('test_df', test.shape)

    # filter non-existing test images
    exists = lambda img: os.path.exists(f'{test_dir}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')
    test = test.loc[test.id.apply(exists)]
    print('test_df after filtering', test.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train.landmark_id = label_encoder.transform(train.landmark_id)

    train_dataset = ImageDataset(train, train_dir, mode='train')
    valid_dataset = ImageDataset(train, train_dir, mode='train')
    # test_dataset = ImageDataset(test, test_dir, mode='test')

    # Split
    valid_size = 0.1
    shuffle = True
    pin_memory = False
    random_seed = 300
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                          shuffle=False, num_workers=4, drop_last=True)

    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                         shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, label_encoder, num_classes


def radam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters,
                      lr=lr,
                      betas=betas,
                      eps=eps,
                      weight_decay=weight_decay)

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res

# class EfficientNetEncoderHead(nn.Module):
#     def __init__(self, depth, num_classes):
#         super(EfficientNetEncoderHead, self).__init__()
#         self.depth = depth
#         self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.output_filter = self.base._fc.in_features
#         self.classifier = nn.Linear(self.output_filter, num_classes)
#     def forward(self, x):
#         x = self.base.extract_features(x)
#         x = self.avg_pool(x).squeeze(-1).squeeze(-1)
#         x = self.classifier(x)
#         return x


def train_step(train_loader,
               model,
               criterion,
               optimizer,
               epoch,
               lr_scheduler):
    print(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    print(f'total batches: {num_steps}')

    end = time.time()
    lr = None

    for i, data in enumerate(train_loader):
        input_ = data['image']
        target = data['target']
        # print(target)
        batch_size, _, _, _ = input_.shape

        output = model(input_.cuda())
        loss = criterion(output, target.cuda())
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                  f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                  + str(lr))

    print(f' * average GAP on train {avg_score.avg:.4f}')


def generate_submission(test_loader, model, label_encoder):
    sample_sub = pd.read_csv('/home/jingshuai/桌面/input/landmark-recognition-2020/sample_submission.csv')

    predicts_gpu, confs_gpu, _ = inference(test_loader, model)
    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs))

    sub = test_loader.dataset.df
    def concat(label: np.ndarray, conf: np.ndarray) -> str:
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv('submission.csv')

def inference(test_loader, model):
    model.eval()
    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []
    exactly_num = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if test_loader.dataset.mode != 'test':
                input_, target = data['image'], data['target']
            else:
                input_, target = data['image'], data['target']

            output = model(input_.cuda())
            output = activation(output)

            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)
            all_confs.append(confs)
            all_predicts.append(predicts)
            # print('_________________')
            # print(target.tolist()[0])
            a = predicts.tolist()[0]
            print(str(target.tolist()[0])+'\t'+str(predicts.tolist()[0][0]))
            if target.tolist()[0] in a:
                exactly_num+=1
            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    print(exactly_num/len(all_confs))
    return predicts, confs, targets


if __name__ == '__main__':
    global_start_time = time.time()
    train_loader, test_loader, label_encoder, num_classes = load_data(train, test, train_dir, test_dir)

    # model = ResModel(num_classes)
    model = SE_VGG(num_classes)
    # model.load_state_dict(torch.load('vgg.pth'))
    # model = models.vgg16(pretrained=True)
    model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = radam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS, eta_min=1e-6)
    starttime = datetime.now()
    for epoch in range(1, NUM_EPOCHS + 1):
        print('-' * 50)
        train_step(train_loader, model, criterion, optimizer, epoch, scheduler)
    # torch.save(model.state_dict(),'vgg.pth')
    print('inference mode')
    endtime = datetime.now()
    print(endtime - starttime)
    # generate_submission(test_loader,model,label_encoder)
