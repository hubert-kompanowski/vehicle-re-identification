import random
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

import wandb
from src import utils
from src.schedulers.warmup import WarmupLR
from src.utils.metrics import MeanAveragePrecision, RankOne, Visualizator

log = utils.get_logger(__name__)


class SimpleReIdLitModule(LightningModule):
    def __init__(
        self,
        optimizer_options,
        backbone: str = "resnet18",
        stage=None,
        checkpoint_path=None,
        map_at_k=100,
    ):
        super().__init__()
        self.learning_rate = optimizer_options["lr"]

        self.save_hyperparameters(logger=False)

        if backbone == "resnet18":
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone == "resnet50":
            self.net = resnet50(weights=ResNet50_Weights.DEFAULT)

        # fine tune
        if stage is not None and stage == "second":

            checkpoint = torch.load(checkpoint_path)

            print(f"Loading model state dict from {checkpoint_path}")
            self.load_state_dict(checkpoint["state_dict"])

            count = 0
            for layer in self.net.children():
                if count <= 5:
                    for param in layer.parameters():
                        param.requires_grad = False
                count += 1

        self.optimizer_options = optimizer_options

        self.criterion = torch.nn.TripletMarginLoss()

        self.test_mAP = MeanAveragePrecision(k=map_at_k)
        self.test_rank_one = RankOne()
        # self.test_visualize = Visualizator()

    def forward(self, x: torch.Tensor):
        return self.net(x.float())

    def step(self, batch: Any):
        x, _ = batch
        anchor_logits = self.forward(x[:, 0, :, :, :])
        positive_logits = self.forward(x[:, 1, :, :, :])
        negative_logits = self.forward(x[:, 2, :, :, :])

        loss = self.criterion(anchor_logits, positive_logits, negative_logits)
        return loss

    def embed(self, batch: Any):
        x, y = batch
        embeddings = self.forward(x)
        return x, embeddings, y

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = sum(output["loss"] for output in outputs) / len(outputs)
        wandb.log({"val_loss_accumulated": loss})

    def test_step(self, batch: Any, batch_idx: int):
        images, embedding, vehicle_id = self.embed(batch)
        self.test_mAP(embedding, vehicle_id)
        self.test_rank_one(embedding, vehicle_id)
        # self.test_visualize(images, embedding, vehicle_id)

        return {"embedding": embedding, "vehicle_id": vehicle_id}

    def test_epoch_end(self, outputs: List[Any]):
        mAP = self.test_mAP.compute_final()
        rank_one = self.test_rank_one.compute_final()
        self.log("test_mAP", mAP, on_step=False, on_epoch=True)
        self.log("test_rank-1", rank_one, on_step=False, on_epoch=True)

        # for i in random.sample(range(1, 200), 2):
        #     images, vids = self.test_visualize.get_images(i, n=5)
        #     wandb.log({"images": wandb.Image(images, caption=', '.join([str(int(item.item())) for item in vids]))})

    def on_epoch_start(self):
        self.log("epoch", self.current_epoch)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

    def on_epoch_end(self):
        self.test_mAP.reset()
        self.test_rank_one.reset()
        # self.test_visualize.reset()

    def configure_optimizers(self):
        log.info(
            f"Chosen optimizer: {self.optimizer_options['optimizer']}, "
            f"lr: {self.learning_rate}, scheduler: {self.optimizer_options['scheduler']}"
        )

        if self.optimizer_options["optimizer"] == "adam":
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        elif self.optimizer_options["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Bad optimizer chosen")

        if self.optimizer_options.scheduler is not None and self.optimizer_options.scheduler.apply:
            if self.optimizer_options.scheduler.type == "WarmupLR":
                scheduler = WarmupLR(
                    optimizer,
                    max_lr=self.optimizer_options.scheduler.max_lr,
                    num_epochs=self.optimizer_options.scheduler.num_epochs,
                )
            elif self.optimizer_options.scheduler.type == "OneCycleLR":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.optimizer_options.scheduler.max_lr,
                    epochs=self.optimizer_options.scheduler.num_epochs,
                    steps_per_epoch=1,
                )
            elif self.optimizer_options.scheduler.type == "CyclicLR":
                scheduler = CyclicLR(
                    optimizer,
                    base_lr=self.learning_rate,
                    max_lr=self.optimizer_options.scheduler.max_lr,
                    step_size_up=5,
                    mode="exp_range",
                    gamma=0.92,
                )
            else:
                raise Exception("Bad scheduler chosen")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                },
            }

        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_options["weight_decay"],
        )
