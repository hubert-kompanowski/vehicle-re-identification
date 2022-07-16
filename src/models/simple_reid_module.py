import random
from typing import Any, List

import torch
import wandb
from pytorch_lightning import LightningModule
from torchvision.models import resnet18, ResNet18_Weights

from src.utils.metrics import MeanAveragePrecision, RankOne, Visualizator


class SimpleReIdLitModule(LightningModule):
    def __init__(self, lr: float = 0.001, weight_decay: float = 0.0005):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.criterion = torch.nn.TripletMarginLoss()

        self.test_mAP = MeanAveragePrecision()
        self.test_rank_one = RankOne()
        self.test_visualize = Visualizator()

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
        pass

    def test_step(self, batch: Any, batch_idx: int):
        images, embedding, vehicle_id = self.embed(batch)
        self.test_mAP(embedding, vehicle_id)
        self.test_rank_one(embedding, vehicle_id)
        self.test_visualize(images, embedding, vehicle_id)

        return {"embedding": embedding, "vehicle_id": vehicle_id}

    def test_epoch_end(self, outputs: List[Any]):
        mAP = self.test_mAP.compute_final()
        rank_one = self.test_rank_one.compute_final()
        self.log("test/mAP", mAP, on_step=False, on_epoch=True)
        self.log("test/rank-1", rank_one, on_step=False, on_epoch=True)

        for i in random.sample(range(1, 200), 10):
            images, vids = self.test_visualize.get_images(i, n=8)
            wandb.log({"images": wandb.Image(images, caption=', '.join([str(int(item.item())) for item in vids]))})

    def on_epoch_end(self):
        self.test_mAP.reset()
        self.test_rank_one.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
