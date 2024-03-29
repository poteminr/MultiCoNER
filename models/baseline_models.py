import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoModel
from torchcrf import CRF


class Bert(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 label_to_id: dict,
                 ):
        super(Bert, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_model, num_labels=self.target_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return output


class BertCRF(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 label_to_id: dict,
                 dropout_rate: float = 0.1,
                 ):
        super(BertCRF, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)

        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)
        self.crf = CRF(self.target_size, batch_first=True)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = input_ids.size(0)
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded_text_input = self.dropout(embedded_text_input)
        token_scores = self.feedforward(embedded_text_input)
        loss, output_tags = self.apply_crf(token_scores, labels, attention_mask, batch_size=batch_size)
        return loss, output_tags, token_scores

    def apply_crf(self, token_scores, labels, attention_mask, batch_size):
        loss = -self.crf(emissions=token_scores, tags=labels, mask=attention_mask) / batch_size
        tags = self.crf.decode(emissions=token_scores, mask=attention_mask)
        return loss, tags


class BertBiLstmCRF(nn.Module):
    def __init__(self,
                 encoder_model: str,
                 label_to_id: dict,
                 dropout_rate: float = 0.1,
                 ):
        super(BertBiLstmCRF, self).__init__()
        self.tag_to_id = label_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.target_size = len(self.id_to_tag)

        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(self.encoder.config.hidden_size, self.encoder.config.hidden_size // 2,
                              dropout=dropout_rate, batch_first=True,
                              bidirectional=True)
        self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)
        self.crf = CRF(self.target_size, batch_first=True)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = input_ids.size(0)
        embedded_text_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded_text_input = self.dropout(embedded_text_input)
        lstm_output, _ = self.bilstm(embedded_text_input)
        token_scores = self.feedforward(lstm_output)
        loss, output_tags = self.apply_crf(token_scores, labels, attention_mask, batch_size=batch_size)
        return loss, output_tags, token_scores

    def apply_crf(self, token_scores, labels, attention_mask, batch_size):
        loss = -self.crf(emissions=token_scores, tags=labels, mask=attention_mask) / batch_size
        tags = self.crf.decode(emissions=token_scores, mask=attention_mask)
        return loss, tags
