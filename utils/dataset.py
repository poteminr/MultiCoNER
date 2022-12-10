import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
from log import logger
import warnings
from utils.reader_utils import get_ner_reader
from utils.tagset import get_tagset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 max_instances: int = -1,
                 max_length: int = 50,
                 encoder_model: str = 'cointegrated/rubert-tiny2',
                 viterbi_algorithm: bool = True,
                 label_pad_token_id: int = -100
                 ):

        self.max_instances = max_instances
        self.max_length = max_length
        self.label_to_id = get_tagset()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.encoder_model = encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model, model_max_length=self.max_length)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        if viterbi_algorithm:
            self.label_pad_token_id = self.pad_token_id
        else:
            self.label_pad_token_id = label_pad_token_id

        self.instances = []
        self.sentences_words = []
        self.read_data(file_path)

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data: str):
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

    # function from huggingface Token Classification ipynb
    # Set label for all tokens and -100 for padding and special tokens
    def tokenize_and_align_labels(self, tokenized_inputs, tags, label_all_tokens=True):
        previous_word_idx = None
        label_ids = []
        for word_idx in tokenized_inputs.word_ids():
            if word_idx is None:
                label_ids.append(self.label_pad_token_id)

            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[self.typos_correction(tags[word_idx])])
            else:
                label_ids.append(self.label_to_id[self.typos_correction(
                    tags[word_idx])] if label_all_tokens else self.label_pad_token_id)
            previous_word_idx = word_idx

        return label_ids

    @staticmethod
    def typos_correction(label: str) -> str:
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


class SiameseDataset(CoNLLDataset):
    def __init__(self,
                 file_path: str,
                 max_instances: int = -1,
                 max_length: int = 50,
                 encoder_model: str = 'cointegrated/rubert-tiny2',
                 viterbi_algorithm: bool = True,
                 label_pad_token_id: int = -100,
                 max_pairs: int = 10000,
                 identical_entities_prob: float = 0.3
                 ):
        super(SiameseDataset, self).__init__(file_path, max_instances, max_length,
                                             encoder_model, viterbi_algorithm, label_pad_token_id)

        self.max_pairs = max_pairs
        self.identical_entities_prob = identical_entities_prob

        self.paired_instances = []
        self.entities_in_data = {}
        self.parse_entities_in_data()
        self.entities = list(self.entities_in_data.keys())
        self.create_pairs()

    def __getitem__(self, item):
        return self.paired_instances[item]

    def __len__(self):
        return len(self.paired_instances)

    def parse_entities_in_data(self):
        for sample_index, (input_ids, labels, _) in enumerate(self.instances):
            previous_label = ''
            for idx, label_id in enumerate(labels):
                label = self.id_to_label[label_id.item()]
                if label.startswith('B-') and label != previous_label:
                    if label not in self.entities_in_data.keys():
                        self.entities_in_data[label] = []
                    self.entities_in_data[label].append((sample_index, idx))
                previous_label = label

    def create_pairs(self):
        used_pairs = set()
        for _ in range(self.max_pairs):
            first_entity = np.random.choice(self.entities)
            idx = self.entities.index(first_entity)
            if np.random.random() > self.identical_entities_prob:
                second_entity = np.random.choice(self.entities[:idx] + self.entities[idx + 1:])
            else:
                second_entity = first_entity

            first_sample, second_sample = self.choice_two_pairs(first_entity, second_entity)

            if (first_sample, second_sample) in used_pairs:
                for _ in range(100):
                    first_sample, second_sample = self.choice_two_pairs(first_entity, second_entity)

            if (first_sample, second_sample) in used_pairs or first_sample == second_sample:
                warning = f'The pair {first_entity}-{second_entity} is not found.'
                warnings.warn(warning)
            else:
                first_input_ids, first_token_mask, first_attention_mask = self.parse_sample(first_entity, first_sample)
                second_input_ids, second_token_mask, second_attention_mask = self.parse_sample(second_entity, second_sample)

                pair_target = float(first_entity == second_entity)
                self.paired_instances.append((
                    [first_input_ids, second_input_ids],
                    [first_token_mask, second_token_mask],
                    [first_attention_mask, second_attention_mask], pair_target))

                used_pairs.add((first_sample, second_sample))
                used_pairs.add((second_sample, first_sample))
        logger.info('Finished creating {:d} pairs.'.format(len(self.paired_instances)))

    def choice_two_pairs(self, first_entity, second_entity):
        rng = np.random.default_rng()
        first_sample = tuple(rng.choice(self.entities_in_data[first_entity]))
        second_sample = tuple(rng.choice(self.entities_in_data[second_entity]))
        for _ in range(100):
            second_sample = tuple(rng.choice(self.entities_in_data[second_entity]))
            if first_sample != second_sample:
                break
        return first_sample, second_sample

    def parse_sample(self, entity, sample):
        sample_index, start_idx = sample
        input_ids, labels, attention_mask = self.instances[sample_index]
        token_mask = np.zeros_like(labels)
        for idx, label in enumerate(labels[start_idx:]):
            if self.id_to_label[label.item()] not in [entity, entity.replace('B', 'I')]:
                break
            token_mask[start_idx+idx] = 1
        return input_ids, token_mask, attention_mask
