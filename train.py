import torch
from utils.dataset import CoNLLDataset, get_dataloader
from transformers import set_seed
from datasets import load_metric
from model.baseline_model import BaselineModel
from torch.optim import AdamW
import numpy as np
import os
import random
from tqdm import tqdm


def seed_everything(seed: int):
    """Seeds and fixes every possible random state."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def compute_metrics(predictions, labels, id_to_label, detailed_output=False, viterbi_algorithm=True):
    predictions = predictions.detach().numpy()
    metric = load_metric("seqeval")
    if not viterbi_algorithm:
        predictions = np.argmax(predictions, axis=2)
        labels = labels.detach().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if detailed_output:
        return results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def train(
        model,
        dataloader,
        epochs: int = 5,
        lr: float = 1e-5,
        betas: tuple = (0.9, 0.95),
        clip_gradients: bool = True,
        grad_norm_clip: float = 1,

) -> None:
    seed_everything(1007)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        average_loss = 0
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for it, (input_ids, labels, attention_mask) in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            if model.viterbi_algorithm:
                loss, output_tags = model(input_ids, labels, attention_mask)
            else:
                output = model(input_ids, labels, attention_mask)
                loss = output['loss']

            average_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        metrics = model.span_f1.get_metric(True)
        average_loss /= len(dataloader)
        print(average_loss, metrics)


if __name__ == "__main__":
    dataset = CoNLLDataset('train_dev/uk-train.conll', label_pad_token_id=None)
    baseline_model = BaselineModel(encoder_model=dataset.encoder_model, label_to_id=dataset.label_to_id)
    train_loader = get_dataloader(dataset=dataset, batch_size=32, num_workers=10)
    train(model=baseline_model, dataloader=train_loader, epochs=10)
