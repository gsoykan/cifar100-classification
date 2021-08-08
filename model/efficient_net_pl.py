import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
from efficientnet_pytorch import EfficientNet

from homework.model.efficient_net.efficient_net_factory import EfficientNetType, EfficientNetFactory


class EfficientNetPL(pl.LightningModule):
    def __init__(self,
                 efficient_net_type: EfficientNetType,
                 output_size: int,
                 lr,
                 use_package_implementation: bool = False):
        super(EfficientNetPL, self).__init__()
        self.save_hyperparameters()
        if not use_package_implementation:
            self.inner_model = EfficientNetFactory.get_efficient_net(efficient_net_type,
                                                                     output_size)
        else:
            # self.inner_model = EfficientNet.from_name('efficientnet-b0',
            #                                          num_classes=output_size)
            self.inner_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=output_size)

    def forward(self, x):
        return self.inner_model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
