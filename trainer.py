import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.xnli_dataset import XNLIDataset
from utils.utils import get_acc


class SiameseTrainer:
    def __init__(
        self,
        device,
        model,
        loss,
        optimizer,
        train_batch_size=10,
        dev_batch_size=4,
        max_epoch=10,
        validation_frequency=1,
        train_log_frequency=500,
    ):
        self.device = device
        self.model = model

        self.dev_batch_size = dev_batch_size
        self.train_batch_size = train_batch_size

        train_dataset = XNLIDataset(
            self.train_model.sentence_embedder,
            tsv_file_path="french_XNLI/multinli.train.fr.tsv",
        )
        dev_dataset = XNLIDataset(
            self.train_model.sentence_embedder,
            tsv_file_path="french_XNLI/xnli.dev.fr.tsv",
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0
        )
        self.dev_dataloader = DataLoader(
            dev_dataset, batch_size=dev_batch_size, num_workers=0
        )

        self.loss = loss
        self.optimizer = optimizer

        self.best_validation_accuracy = 0

        self.epoch = 0
        self.validation_frequency = validation_frequency
        self.train_log_frequency = train_log_frequency
        self.max_epoch = max_epoch

        self.writer = SummaryWriter()
        self.save_dir = "saved_models"

    def get_targets(self, labels):
        def get_target(label):
            if label == "contradiction":
                return torch.tensor([0])
            elif label == "neutral":
                return torch.tensor([1])
            else:
                return torch.tensor([2])

        return torch.tensor(list(map(get_target, labels))).to(self.device)

    def get_targets(self, labels):
        def get_target(label):
            if label == "contradictory":
                return torch.tensor([0])
            elif label == "neutral":
                return torch.tensor([1])
            else:
                return torch.tensor([2])

        return torch.tensor(list(map(get_target, labels))).to(self.device)

    def train(self):
        print("==> Start training!")
        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch

            # training
            self.train_epoch()

            # validation
            if epoch % self.validation_frequency == 0:
                self.validate()

        print(f"=> Train finished | best_mse: {self.best_validation_accuracy}")

    def train_epoch(self):
        self.model.train()
        current_loss = 0
        train_accuracy = 0
        epoch_start = time.time()

        for step, sample_batched in enumerate(self.train_dataloader):

            with torch.set_grad_enabled(True):
                output = self.model(sample_batched)
                loss = self.loss(output, self.get_targets(sample_batched["label"]))
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                current_loss += loss.detach()
                train_accuracy += get_acc(
                    output, self.get_targets(sample_batched["label"])
                )

            if step % self.train_log_frequency == 0 and step > 0:
                current_loss /= self.train_log_frequency
                train_accuracy /= self.train_log_frequency

                self.writer.add_scalar(
                    "Train/loss", current_loss, self.epoch * len(self.train_dataloader)
                )
                self.writer.add_scalar(
                    "Train/accuracy",
                    train_accuracy,
                    self.epoch * len(self.train_dataloader),
                )

                print(
                    f"> epoch: {self.epoch} | step: {step} | loss: {loss} | train_accuracy: {train_accuracy} | epoch_training_time: {time.time() - epoch_start}"
                )

                current_loss = 0
                train_accuracy = 0

        print(f"=> Train epoch {self.epoch} finished in {time.time() - epoch_start} ms")

    def validate(self):
        validation_start = time.time()
        validation_accuracy = 0
        self.model.eval()  # Set model to evaluate mode

        # Iterate over data.
        for step, sample_batched in enumerate(self.dev_dataloader):
            with torch.set_grad_enabled(False):
                output = self.model(sample_batched)
                validation_accuracy += get_acc(
                    output, self.get_targets_dev(sample_batched["label"])
                )

        validation_accuracy /= len(self.dev_dataloader)
        self.writer.add_scalar("Validation/accuracy", validation_accuracy, self.epoch)

        print(
            f"=> Validation epoch {self.epoch} | validation_accuracy: {validation_accuracy} | validation_time: {time.time() - validation_start} ms"
        )

        if validation_accuracy > self.best_validation_accuracy:
            torch.save(self.model, f"{self.save_dir}/{self.epoch}_siamese.pth")
            print("====> New best model saved!")
