import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from log import logger
from utils.reader_utils import get_ner_reader
from utils.tagset import get_tagset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLDataset(Dataset):
    def __init__(self,
                 file_path,
                 max_instances=-1,
                 max_length=50,
                 target_vocab=None,
                 encoder_model='cointegrated/rubert-tiny2',
                 label_pad_token_id=-100
                 ):

        self.max_instances = max_instances
        self.max_length = max_length
        self.label_to_id = get_tagset() if target_vocab is None else target_vocab
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.encoder_model = encoder_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model, model_max_length=self.max_length)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_pad_token_id = self.pad_token_id if label_pad_token_id is None else label_pad_token_id
        self.instances = []
        self.sentences_words = []

        self.read_data(file_path)

    def get_target_size(self):
        if self.label_pad_token_id == -100:
            return len(set(self.label_to_id.values())) + 1
        else:
            return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self.max_instances != -1 and instance_idx > self.max_instances:
                break

            sentence_words, tags = fields[0], fields[-1]
            tokenized_inputs = self.tokenizer(sentence_words, truncation=True, is_split_into_words=True)

            input_ids = torch.tensor(tokenized_inputs['input_ids'], dtype=torch.long)
            labels = torch.tensor(self.tokenize_and_align_labels(tokenized_inputs, tags))
            attention_mask = torch.tensor(tokenized_inputs['attention_mask'], dtype=torch.bool)

            self.instances.append((input_ids, labels, attention_mask))
            self.sentences_words.append([sentence_words, tags])
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def tokenize_and_align_labels(self, tokenized_inputs, tags, label_all_tokens=True):
        previous_word_idx = None
        label_ids = []
        for word_idx in tokenized_inputs.word_ids():
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(self.label_pad_token_id)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[self.typos_correction(tags[word_idx])])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(self.label_to_id[self.typos_correction(
                    tags[word_idx])] if label_all_tokens else self.label_pad_token_id)
            previous_word_idx = word_idx

        return label_ids

    @staticmethod
    def typos_correction(label):
        if label[-4:] == 'Corp':
            return label[:-4] + "CORP"

        return label

    def data_collator(self, batch):
        batch_ = list(zip(*batch))
        input_ids, labels, attention_masks = batch_[0], batch_[1], batch_[2]

        max_length_in_batch = max([len(token) for token in input_ids])
        input_ids_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
            self.pad_token_id)
        labels_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
            self.label_pad_token_id)
        attention_masks_tensor = torch.zeros(size=(len(input_ids), max_length_in_batch), dtype=torch.bool)

        for i in range(len(input_ids)):
            tokens_ = input_ids[i]
            seq_len = len(tokens_)

            input_ids_tensor[i, :seq_len] = tokens_
            labels_tensor[i, :seq_len] = labels[i]
            attention_masks_tensor[i, :seq_len] = attention_masks[i]

        return input_ids_tensor, labels_tensor, attention_masks_tensor


def get_dataloader(dataset, batch_size, num_workers=1):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=dataset.data_collator, num_workers=num_workers)
