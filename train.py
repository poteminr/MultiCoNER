import torch
from torch.utils.data import DataLoader
from utils.dataset import CoNLLDataset
from transformers import set_seed
from datasets import load_metric
from model.baseline_model import BaselineModel
from torch.optim import AdamW
import numpy as np
import os
import random
from tqdm import tqdm


class TrainerConfig:
    epochs = 10
    lr = 1e-5
    batch_size = 32
    betas = (0.9, 0.95)
    clip_gradients = True
    grad_norm_clip = 1.0
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset: CoNLLDataset, test_dataset: CoNLLDataset, config: TrainerConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.viterbi_algorithm = model.viterbi_algorithm
        self.label_pad_token_id = self.train_dataset.label_pad_token_id
        self.metric = load_metric("seqeval")

    def train(self):
        self.seed_everything(1007)
        model = self.model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.config.lr, betas=self.config.betas)
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.batch_size,
                                  collate_fn=self.train_dataset.data_collator, num_workers=self.config.num_workers)

        for epoch in range(self.config.epochs):
            average_loss = 0
            model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for it, (input_ids, labels, attention_mask) in pbar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)

                loss, output = model(input_ids, labels, attention_mask)
                average_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                if self.config.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_gradients)
                optimizer.step()
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

            metrics = self.compute_metrics(output, labels, self.train_dataset.label_to_id)
            average_loss /= len(train_loader)
            print(average_loss, metrics)

    def compute_metrics(self, predictions, labels, id_to_label, detailed_output=False):
        predictions = predictions.detach().numpy()
        if not self.viterbi_algorithm:
            predictions = np.argmax(predictions, axis=2)
            labels = labels.detach().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != self.label_pad_token_id]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != self.label_pad_token_id]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
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

# if __name__ == "__main__":
# dataset = CoNLLDataset('train_dev/uk-train.conll', label_pad_token_id=None)
# baseline_model = BaselineModel(encoder_model=dataset.encoder_model, label_to_id=dataset.label_to_id)
# train_loader = get_dataloader(dataset=dataset, batch_size=32, num_workers=10)
# train(model=baseline_model, dataloader=train_loader, epochs=10)
