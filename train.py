import torch
from torch.utils.data import DataLoader
from utils.dataset import CoNLLDataset
from transformers import set_seed
from evaluate import load
from model.baseline_model import BaselineModel
from torch.optim import AdamW
import numpy as np
import os
import random
from tqdm import tqdm
from typing import Optional
from options import train_options


class TrainerConfig:
    epochs = 10
    lr = 1e-5
    batch_size = 64
    betas = (0.9, 0.95)
    clip_gradients = True
    grad_norm_clip = 1.0
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset: CoNLLDataset, test_dataset: Optional[CoNLLDataset], config: TrainerConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.id_to_label = self.train_dataset.id_to_label
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.viterbi_algorithm = model.viterbi_algorithm
        self.label_pad_token_id = self.train_dataset.label_pad_token_id
        self.metric = load("seqeval")

    def train(self):
        self.seed_everything(1007)
        model = self.model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.config.lr, betas=self.config.betas)
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.batch_size,
                                  collate_fn=self.train_dataset.data_collator, num_workers=self.config.num_workers)

        for epoch in range(self.config.epochs):
            average_loss = 0
            average_f1 = 0
            model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
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

                average_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                if self.config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_gradients)
                optimizer.step()
                metrics = self.compute_metrics(predictions=output, labels=labels, detailed_output=False)
                f1 = metrics['f1']
                average_f1 += f1
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. f1 {f1:.8f}.")

            average_loss /= len(train_loader)
            average_f1 /= len(train_loader)
            print(average_loss, average_f1)

    def compute_metrics(self, predictions, labels, detailed_output=False):
        labels = labels.detach().cpu().numpy()
        if not self.viterbi_algorithm:
            predictions = predictions.detach().numpy()
            predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
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
        """Seeds and fixes every possible random state."""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        set_seed(seed)


if __name__ == "__main__":
    arguments = train_options()
    dataset = CoNLLDataset(file_path='train_dev/uk-train.conll', viterbi_algorithm=arguments.viterbi)
    baseline_model = BaselineModel(encoder_model=dataset.encoder_model, label_to_id=dataset.label_to_id, viterbi_algorithm=arguments.viterbi)
    config = TrainerConfig()
    trainer = Trainer(baseline_model, train_dataset=dataset, test_dataset=None, config=config)
    trainer.train()
