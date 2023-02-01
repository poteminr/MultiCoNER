import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from torchcrf import CRF


def distance_based_probability(x: torch.tensor, y: torch.tensor, margin: float = 1.0) -> float:
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    euclidean_distance = torch.square(torch.sum(torch.square(x - y), dim=1, keepdim=True))
    p = (1.0 + np.exp(-margin)) / (1.0 + torch.exp(euclidean_distance) - margin)
    return p


def masked_mean_pooling(data_tensor: torch.tensor, mask: torch.tensor, dim: int) -> torch.tensor:
    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, 0)

    nominator = torch.sum(data_tensor, dim=dim)
    denominator = torch.sum(mask.type(nominator.type()), dim=dim)
    return nominator / denominator


class CoBert(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 label_to_id: dict,
                 dropout_rate: float = 0.1,
                 ):
        super(CoBert, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)

        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded_text_input = self.dropout(embedded_text_input)
        return embedded_text_input


class CoBertCRF(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 pretrained_encoder_model_path: str,
                 label_to_id: dict,
                 dropout_rate: float = 0.1,
                 ):
        super(CoBertCRF, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)
        
        self.encoder = CoBert(encoder_model, label_to_id, dropout_rate)
        if pretrained_encoder_model_path is not None:
            self.encoder.load_state_dict(torch.load(pretrained_encoder_model_path))
        
        self.feedforward = nn.Linear(in_features=self.encoder.encoder.config.hidden_size, out_features=self.target_size)
        self.crf = CRF(self.target_size, batch_first=True)

    def get_token_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_scores = self.feedforward(embedded_text_input)
        return token_scores
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = input_ids.size(0)
        token_scores = self.get_token_scores(input_ids=input_ids, attention_mask=attention_mask)
        loss, output_tags = self.apply_crf(token_scores, labels, attention_mask, batch_size=batch_size)
        return loss, output_tags, token_scores

    def apply_crf(self, token_scores, labels, attention_mask, batch_size):
        loss = -self.crf(emissions=token_scores, tags=labels, mask=attention_mask) / batch_size
        tags = self.crf.decode(emissions=token_scores, mask=attention_mask)
        return loss, tags
    
    def predict_tags(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        token_scores = self.get_token_scores(input_ids=input_ids, attention_mask=attention_mask)
        tags = self.crf.decode(emissions=token_scores, mask=attention_mask)
        return tags
    