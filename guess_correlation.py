import os
import random
import argparse
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2

from CorrelationDataset import CorrelationDataset
from Trainer import Trainer
from models import get_model_optimizer


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
        "--load_model_from",
        type=str,
        default=None,
        help="Load a model to evaluate or resume training.",
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
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="Whether to evaluate the model.",
    )
    parser.add_argument(
        "--do_eval",
        type=bool,
        default=True,
        help="Whether to evaluate the model.",
    )

    args = parser.parse_args()

    assert args.optimizer in [
        "sgdm",
        "adam",
        "adamw",
    ], "--optimizer only supports {sgdm, adam, adamw}."

    assert args.backbone in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "vit_b_16",
    ], "--optimizer only supports {sgdm, adam, adamw}."

    return args


def main(args):
    # Reproduct
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Read & Prepare Data
    df = pd.read_csv(args.annotation_file)

    trainval_df, test_df = train_test_split(
        df, test_size=args.test_ratio if args.do_eval else 0, random_state=args.seed
    )

    train_df, val_df = train_test_split(
        trainval_df, test_size=0.2, random_state=args.seed
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
    test_dataset = CorrelationDataset(test_df, args.image_dir, transforms)

    with wandb.init(project=args.project_name, name=args.run_name, config=vars(args)):

        model = get_model_optimizer(
            backbone=args.backbone,
            use_tanh=args.use_tanh,
            optimizer_type=args.optimizer,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint=args.load_model_from,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        trainer = Trainer(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            criterion=nn.MSELoss(reduction="mean"),
            optimizer=optimizer,
            scheduler=lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epoch, eat_min=0
            ),  # lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            device="cuda" if torch.cuda.is_available() else "cpu",
            tolerance=args.tolerance,
            save_path=args.output_dir,
        )

        if args.do_train:
            trainer.train(num_epoch=args.num_epoch)

        if args.do_eval:
            rmse, mse, mae, r2 = trainer.eval()
            print(f"The evaluation result on testing set:")
            print(f"RMSE: {rmse}")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"R2 Score: {r2}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
