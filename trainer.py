import torch
from torch.utils.data import DataLoader
from utils.dataset import CoNLLDataset
from transformers import set_seed
from evaluate import load
from torch.optim import AdamW
import numpy as np
import os
import random
from tqdm import tqdm
from typing import Optional
import wandb


class TrainerConfig:
    epochs = 20
    lr = 1e-4
    batch_size = 32
    betas = (0.9, 0.95)
    clip_gradients = False
    grad_norm_clip = 10
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, config: TrainerConfig, train_dataset: CoNLLDataset,
                 val_dataset: Optional[CoNLLDataset] = None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.id_to_label = self.train_dataset.id_to_label
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.viterbi_algorithm = model.viterbi_algorithm
        self.label_pad_token_id = self.train_dataset.label_pad_token_id
        self.metric = load("seqeval")
        self.seed = 1007

    def create_dataloader(self, dataset: CoNLLDataset):
        return DataLoader(dataset=dataset, batch_size=self.config.batch_size,
                          collate_fn=dataset.data_collator, num_workers=self.config.num_workers)

    def perform_epoch(self, epoch, model, optimizer, loader, train_mode: bool):
        average_loss = 0
        average_f1 = 0
        if train_mode:
            model.train()
            text = 'train'
            newline = ''
        else:
            model.eval()
            text = 'val'
            newline = '\n'

        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (input_ids, labels, attention_mask) in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if self.viterbi_algorithm:
                result = model(input_ids, labels, attention_mask)
                loss, output = result[0], result[1]
            else:
                result = model(input_ids, labels, attention_mask)
                loss, output = result['loss'], result['logits']

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if self.config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_gradients)
                optimizer.step()

            metrics = self.compute_metrics(predictions=output, labels=labels, detailed_output=False)
            average_loss += loss.item()
            average_f1 += metrics['f1']

            pbar.set_description(
                f"epoch {epoch + 1} iter {it} | {text}_loss: {loss.item():.5f}. {text}_f1: {metrics['f1']:.8f}.")

        average_loss /= len(loader)
        average_f1 /= len(loader)
        wandb.log({
            f'{text}/loss': average_loss,
            f'{text}/f1': average_f1,
        },
            step=epoch + 1)
        print(f"{text}_loss: {average_loss}", f"{text}_f1: {average_f1}{newline}")

    def train(self):
        self.seed_everything(self.seed)
        model = self.model.to(self.device)
        wandb.watch(model)
        optimizer = AdamW(model.parameters(), lr=self.config.lr, betas=self.config.betas)
        train_loader = self.create_dataloader(self.train_dataset)
        if self.val_dataset is not None:
            val_loader = self.create_dataloader(self.val_dataset)

        for epoch in range(self.config.epochs):
            self.perform_epoch(epoch, model, optimizer, train_loader, train_mode=True)
            if self.val_dataset is not None:
                self.perform_epoch(epoch, model, optimizer, val_loader, train_mode=False)

        wandb.finish()

    def compute_metrics(self, predictions, labels, detailed_output=False):
        labels = labels.detach().cpu().numpy()
        if not self.viterbi_algorithm:
            predictions = predictions.detach().cpu().numpy()
            predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != self.label_pad_token_id]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [self.id_to_label[l] for (p, l) in zip(prediction, label) if l != self.label_pad_token_id]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        if detailed_output:
            return results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        set_seed(seed)
