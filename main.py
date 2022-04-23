import os
from sched import scheduler
import torch
import numpy as np
import pandas as pd
import pretrainedmodels
import torch.nn as nn
import albumentations


from apex import amp

from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationDataLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
#Area under curve 


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)
    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        return out
 
def run(fold):
    training_data_path = "/home/karan/Desktop/Melonoma-Classifier/jpeg/resized_train"
    df = pd.read_csv = "/home/karan/Desktop/Melonoma-Classifier/train_Folds.csv"
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )


    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )


    train_images = df.train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df.valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = True,
        num_workers = 4
    )



    valid_dataset = ClassificationDataLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = valid_bs,
        shuffle = False,
        num_workers = 4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=torch.le-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode = "max"
    )


    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level = "01",
        verbosity = 0
    )


    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            fp16=True
        )
        predictions, valid_loss = Engine.evaluate(
            train_loader,
            model,
            optimizer,
            device            
        )
