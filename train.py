import os
import random
import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
from torchvision.io import read_image

from CorrelationDataset import CorrelationDataset
from Trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model to guess correlation.")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=None,
        help="A csv file containing the files'(images) name and labels.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="The directory of the images",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="The ratio of testing set.",
    )
    parser.add_argument(
        "--resize_to",
        type=int,
        default=150,
        help="The size of input images after resizing.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="The backbone for feature extraction. Select from {resnet18, resnet34, resnet50, resnet101, vit_b_16}",
    )
    parser.add_argument(
        "--use_tanh",
        type=bool,
        default=True,
        help="Whether to use tanh activation as the final layer of the whole model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size used in training and validating.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgdm",
        help="The project name used in wandb. Select from {sgdm, adam, adamw}",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The learning rate used in training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="The regularization coefficient used in training.",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=50,
        help="The number of epochs.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="The maximum number of epochs without improvement. Exceed this tolerance will triger the early stopping mechanism.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="The directory saving the output models and files.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="Guess Correlation",
        help="The project name used in wandb.",
    )
    parser.add_argument(
        "--run_name", type=str, default="baseline", help="The run name used in wandb."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    assert parser.optimizer in [
        "sgdm",
        "adam",
        "adamw",
    ], "--optimizer only supports {sgdm, adam, adamw}."

    assert parser.backbone in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "vit_b_16",
    ], "--optimizer only supports {sgdm, adam, adamw}."

    return parser


def main(args):
    # Reproduct
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Read & Prepare Data
    df = pd.read_csv(args.annotation_file)

    train_df, val_df = train_test_split(
        df, test_size=args.test_ratio, random_state=args.seed
    )

    transforms = v2.Compose(
        [
            v2.Resize(size=(args.resize_to, args.resize_to), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = CorrelationDataset(train_df, args.image_dir, transforms)
    val_dataset = CorrelationDataset(val_df, args.image_dir, transforms)

    print(train_dataset[0])

    with wandb.init(
        project=args.project_name, name=configargs.run_name, config=vars(args)
    ):

        model = get_model(backbone=args.backbone, use_tanh=args.use_tanh)

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        if args.opimizer == "sgdm":

            optimizer = optim.SGD(
                model.parameters(),
                momentum=0.9,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        elif args.opimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        trainer = Trainer(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            criterion=nn.MSELoss(reduction="mean"),
            optimizer=optimizer,
            scheduler=None,  # lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            device="cuda" if torch.cuda.is_available() else "cpu",
            tolerance=args.tolerance,
            save_path=args.output_dir,
        )

        trainer.train(num_epoch=args.num_epoch)

    print("exit")


if __name__ == "__main__":
    args = parse_args()
    main(args)
