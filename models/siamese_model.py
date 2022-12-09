import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel


def distance_based_probability(x: torch.Tensor, y: torch.Tensor, margin: float = 1.0):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    euclidean_distance = torch.square(torch.sum(torch.square(x-y), dim=1, keepdim=True))
    p = (1.0 + np.exp(-margin)) / (1.0 + np.exp(euclidean_distance) - margin)

    return p


class BertBiLstm(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 label_to_id: dict,
                 dropout_rate: float = 0.1,
                 ):
        super(BertBiLstm, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)

        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(self.encoder.config.hidden_size, self.encoder.config.hidden_size // 2,
                              dropout=dropout_rate, batch_first=True,
                              bidirectional=True)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded_text_input = self.dropout(embedded_text_input)
        return embedded_text_input

