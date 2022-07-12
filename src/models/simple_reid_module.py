import pickle
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import LabelRankingAveragePrecision, MaxMetric
from torchvision.models import ResNet50_Weights, resnet50


class SimpleReIdLitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, lr: float = 0.001, weight_decay: float = 0.0005):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.criterion = torch.nn.TripletMarginLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_acc = LabelRankingAveragePrecision()
        # self.val_acc = LabelRankingAveragePrecision()
        # self.test_acc = LabelRankingAveragePrecision()
        #
        # # for logging best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    # def on_train_start(self):
    #     # by default lightning executes validation step sanity checks before training starts,
    #     # so we need to make sure val_acc_best doesn't store accuracy from these checks
    #     self.val_acc_best.reset()

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
        return embeddings, y

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # log train metrics
        # acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}  # , "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # log val metrics
        # acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}  # , "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass
        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        embedding, vehicle_id = self.embed(batch)
        return embedding, vehicle_id
        # return {"embedding": embedding, "vehicle_id": vehicle_id}

    def test_epoch_end(self, outputs: List[Any]):
        with open("test_output.pkl", "wb") as file:
            pickle.dump(outputs, file)
            print("test outputs saved")

        # embeddings = [element[0] for sublist in outputs for element in sublist]
        # vehicle_ids = [element[1] for sublist in outputs for element in sublist]
        # distances = torch.cdist(torch.tensor(embeddings), torch.tensor(embeddings))
        # for vid, dists in zip(vehicle_ids, distances):
        #
        # pass

    # def on_epoch_end(self):
    #     # reset metrics at the end of every epoch
    #     self.train_acc.reset()
    #     self.test_acc.reset()
    #     self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
