import torch
from utils.dataset import CoNLLDataset, get_dataloader
from transformers import set_seed
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


def train(model, dataloader, epochs, lr=1e-5, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if optimizer is not None:
        optimizer = optimizer(model.parameters(), lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        average_loss = 0
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for it, (input_ids, labels, attention_mask) in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            output = model(input_ids, labels, attention_mask)
            loss = output['loss']
            average_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        metrics = model.span_f1.get_metric(True)
        average_loss /= len(dataloader)
        print(average_loss, metrics)


if __name__ == "__main__":
    seed_everything(1007)
    dataset = CoNLLDataset('train_dev/uk-train.conll', label_pad_token_id=None)
    model = BaselineModel(encoder_model=dataset.encoder_model, label_to_id=dataset.label_to_id)
    train_loader = get_dataloader(dataset=dataset, batch_size=32, num_workers=10)
    train(model=model, dataloader=train_loader, epochs=10)
