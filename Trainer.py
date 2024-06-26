import os
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        tolerance,
        save_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tolerance = tolerance
        self.save_path = save_path
        self.best_loss = np.inf
        self.model.to(self.device)

    def _train_one_epoch(self):
        self.model.train()
        loss_record = []

        p_bar = tqdm(self.train_loader, position=0, leave=True)
        p_bar.set_description("Train")

        for step, (images, labels) in enumerate(p_bar):

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            loss_record.append(loss.detach().item())

            p_bar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)

        print(f"Train loss: {mean_train_loss}")

        return mean_train_loss

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_record = []

        p_bar = tqdm(self.val_loader, position=0, leave=True)
        p_bar.set_description("Validate")

        for step, (images, labels) in enumerate(p_bar):

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss_record.append(loss.item())

            p_bar.set_postfix({"loss": loss.item()})

        mean_val_loss = sum(loss_record) / len(loss_record)

        print(f"Validate loss: {mean_val_loss}")

        return mean_val_loss

    def train(self, num_epoch=10):
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        no_improve = 0

        self.model.to(self.device)

        for epoch in range(num_epoch):
            print(f"Epoch [{epoch + 1}/{num_epoch}]")

            train_mean_loss = self._train_one_epoch()
            if self.scheduler is not None:
                wandb.log(
                    {
                        "learning rate": float(self.scheduler.get_lr()[0]),
                    },
                    step=epoch,
                )
                print(f"lr: {self.scheduler.get_lr()[0]}")
                self.scheduler.step()

            val_mean_loss = self._validate()

            wandb.log(
                {
                    "epoch": epoch,
                    "train loss": train_mean_loss,
                    "val loss": val_mean_loss,
                },
                step=epoch,
            )

            if val_mean_loss < self.best_loss:
                self.best_loss = val_mean_loss
                no_improve = 0
                print(f"Saving the model with val loss {val_mean_loss}")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.save_path, "best.pt"),
                )
            else:
                no_improve += 1

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(self.save_path, "last.pt"),
            )

            if no_improve >= self.tolerance:
                print(
                    f"No improvement after {self.tolerance} epochs, so stop training."
                )
                print(f"The best val loss is {val_mean_loss}")
                break

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        output_record = []
        label_record = []

        self.model.to(self.device)

        for images, labels in tqdm(self.test_loader):

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            output_record.extend(outputs.tolist())

            label_record.extend(labels.tolist())

        mse = mean_squared_error(label_record, output_record)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(label_record, output_record)
        r2 = r2_score(label_record, output_record)

        return rmse, mse, mae, r2
