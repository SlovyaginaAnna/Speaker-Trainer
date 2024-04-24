import torch
import torch.nn as nn
from torchvision.models import resnet34
import pytorch_lightning as pl
from sklearn.metrics import fbeta_score
import torch.nn.functional as F
from fastai.vision.all import *

class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    """
    Modified loss function.
    """
    def init(self, eps:float=0.1, **kwargs):
        self.eps = eps
        super().init(thresh=0.2, **kwargs)

    def call(self, inp, targ, **kwargs):
        # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#929222
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().call(inp, targ_smooth, **kwargs)

class CustomResNet(nn.Module):
    """
    Tuned resnet 34 model.
    """
    def __init__(self, num_classes=19):
        """
        Initialize resnet34 model and change last layer.
        :param num_classes: int number of outputs.
        """
        super(CustomResNet, self).__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class CustomModel(pl.LightningModule):
    def __init__(self, model, threshold=0.7, k=4):
        super(CustomModel, self).__init__()
        self.model = model
        self.train_loss_mean = []
        self.train_acc_mean = []
        self.train_k_acc = []
        self.val_loss_mean = []
        self.val_acc_mean = []
        self.val_k_acc = []
        self.k = k
        self.threshold = threshold

    def adversarial_loss(self, y_hat, y):
        """
        Initialize loss function.
        :param y_hat: prediction.
        :param y: real values.
        :return: loss function.
        """
        loss_fn = LabelSmoothingBCEWithLogitsLossFlat()
        return loss_fn(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Step of the training loop.
        :param batch: batch for training.
        :param batch_idx: index of trained batch.
        :return: loss calculated on this step.
        """
        images, attributes = batch
        outputs = self(images)
        loss = self.adversarial_loss(outputs, attributes)
        self.train_loss_mean.append(loss)
        accuracy = self.calculate_accuracy(outputs, attributes)
        k_acc = self.top_k_accuracy(outputs, attributes)
        self.train_acc_mean.append(accuracy)
        self.train_k_acc.append(k_acc)
        print(batch_idx, loss.item(), accuracy, k_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Step of the validation loop.
        :param batch: batch for validation.
        :param batch_idx: index of validation batch.
        :return: dictionary with validation loss and accuracy.
        """
        images, attributes = batch
        outputs = self(images)
        loss = self.adversarial_loss(outputs, attributes)
        self.val_loss_mean.append(loss)
        accuracy = self.calculate_accuracy(outputs, attributes)
        k_acc = self.top_k_accuracy(outputs, attributes)
        self.val_acc_mean.append(accuracy)
        self.val_k_acc.append(k_acc)
        return {"val_loss": loss, "val_accuracy": accuracy, "val_k_acc": k_acc}

    def configure_optimizers(self):
        """
        Initialize optimizer.
        :return: optimizer.
        """
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def on_validation_epoch_end(self):
        """
        Calculate mean values when validation epoch ends.
        """
        print('val epoch end')
        loss = sum(self.val_loss_mean) / len(self.val_loss_mean)
        self.val_loss_mean = []
        acc = sum(self.val_acc_mean) / len(self.val_acc_mean)
        self.val_acc_mean = []
        k_acc = sum(self.val_k_acc) / len(self.val_k_acc)
        self.val_k_acc = []
        self.log("val epoch end loss", loss, prog_bar=True)
        self.log("val epoch end acc", acc, prog_bar=True)
        self.log("val epoch end k acc", k_acc, prog_bar=True)

    def calculate_accuracy(self, outputs, targets):
        """
        Calculate the quality of the model.
        :param outputs: model outputs.
        :param targets: targets: real values.
        :return: float value - accuracy.
        """
        probs = F.softmax(outputs, dim=1)
        binary_mask = (probs >= self.threshold).float()
        accuracy = fbeta_score(binary_mask, targets, beta=2, average='samples')
        return accuracy

    def on_train_epoch_end(self):
        """
        Calculate mean values when trining epoch ends.
        """
        loss = sum(self.train_loss_mean) / len(self.train_loss_mean)
        self.train_loss_mean = []
        acc = sum(self.train_acc_mean) / len(self.train_acc_mean)
        self.train_acc_mean = []
        k_acc = sum(self.train_k_acc) / len(self.train_k_acc)
        self.train_k_acc = []
        self.log("train epoch end loss", loss, prog_bar=True)
        self.log("train epoch end acc", acc, prog_bar=True)
        self.log("train epoch end k acc", k_acc, prog_bar=True)

    def top_k_accuracy(self, outputs, targets):
        """
        Calculate accuracy among k most probable classes.
        :param outputs: model outputs.
        :param targets: real values.
        :return: float value - accuracy.
        """
        topk_values, topk_indices = torch.topk(outputs, self.k, dim=1)
        correct_count = 0
        for i in range(topk_indices.size(0)):
            for j in range(topk_indices.size(1)):
                if targets[i, topk_indices[i, j]] == 1:
                    correct_count += 1
        accuracy = correct_count / (outputs.size(0) * self.k)
        return accuracy