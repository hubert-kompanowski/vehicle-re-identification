import random
from typing import Any, List

import torch
import wandb
from pytorch_lightning import LightningModule
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

from src.schedulers.warmup import WarmupLR
from src.utils.metrics import MeanAveragePrecision, RankOne, Visualizator


class SimpleReIdLitModule(LightningModule):
    def __init__(self, optimizer_options: dict, backbone: str = 'resnet18'):
        super().__init__()
        self.learning_rate = optimizer_options['lr']

        self.save_hyperparameters(logger=False)
        if backbone == 'resnet18':
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet34':
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.optimizer_options = optimizer_options

        self.criterion = torch.nn.TripletMarginLoss()

        self.test_mAP = MeanAveragePrecision()
        self.test_rank_one = RankOne()
        # self.test_visualize = Visualizator()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, _ = batch
        anchor_logits = self.forward(x[0])
        positive_logits = self.forward(x[1])
        negative_logits = self.forward(x[2])
        loss = self.criterion(anchor_logits, positive_logits, negative_logits)
        return loss

    def embed(self, batch: Any):
        x, y = batch
        embeddings = self.forward(x)
        return x, embeddings, y

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        wandb.log({'val_loss_accumulated': loss})

    def test_step(self, batch: Any, batch_idx: int):
        images, embedding, vehicle_id = self.embed(batch)
        self.test_mAP(embedding, vehicle_id)
        self.test_rank_one(embedding, vehicle_id)
        # self.test_visualize(images, embedding, vehicle_id)

        return {"embedding": embedding, "vehicle_id": vehicle_id}

    def test_epoch_end(self, outputs: List[Any]):
        mAP = self.test_mAP.compute_final()
        rank_one = self.test_rank_one.compute_final()
        self.log("test/mAP", mAP, on_step=False, on_epoch=True)
        self.log("test/rank-1", rank_one, on_step=False, on_epoch=True)

        # for i in random.sample(range(1, 200), 2):
        #     images, vids = self.test_visualize.get_images(i, n=5)
        #     wandb.log({"images": wandb.Image(images, caption=', '.join([str(int(item.item())) for item in vids]))})

    def on_epoch_start(self):
        self.log("epoch", self.current_epoch)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_epoch_end(self):
        self.test_mAP.reset()
        self.test_rank_one.reset()
        # self.test_visualize.reset()

    def configure_optimizers(self):
        if self.optimizer_options['optimizer'] == 'adam':
            if self.optimizer_options['scheduler'] is None:
                return torch.optim.Adam(
                    params=self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.optimizer_options['weight_decay']
                )
            else:

                optimizer = torch.optim.Adam(
                    params=self.parameters(),
                    lr=self.learning_rate
                )

                scheduler = WarmupLR(optimizer)

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train_loss",
                        "interval": "epoch"
                    }
                }
