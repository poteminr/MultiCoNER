from torch.utils.data import DataLoader
from utils.tagset import get_tagset
from utils.reader import CoNLLReader, CoNLLUntokenizedReader
import torch

class TrainingData:
    def __init__(self,
                 train_file_path,
                 val_file_path=None,
                 max_instances=-1,
                 max_length=50,
                 encoder_model='cointegrated/rubert-tiny2',
                 batch_size=16):
        
        self.max_instances = max_instances
        self.max_length = max_length
        self.encoder_model = encoder_model

        self.target_vocab = get_tagset()
                
        self.train_data = self.get_reader(train_file_path)
        self.val_data = self.get_reader(val_file_path)
        
        self.pad_token_id = self.train_data.pad_token_id

        if encoder_model is not None:
            self.tag_to_id = self.train_data.get_target_vocab() # equals to self.target_vocab
            self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
            self.target_size = len(self.id_to_tag)
            
        self.batch_size = batch_size
        
    def get_reader(self, file_path):
        if file_path is None:
            return None
                
        if self.encoder_model is None:
            reader = CoNLLUntokenizedReader(
                max_instances=self.max_instances,
                max_length=self.max_length,
                target_vocab=self.target_vocab
                )
        else:
            reader = CoNLLReader(
                max_instances=self.max_instances,
                max_length=self.max_length,
                target_vocab=self.target_vocab,
                encoder_model=self.encoder_model
                )
        
        reader.read_data(file_path)
        return reader
        
    def collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=10)
        return loader
    
    def val_dataloader(self):
        if self.val_data is None:
            return None
        
        loader = DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=10)
        return loader

    

if __name__ == "__main__":
    dataset = TrainingData('train_dev/uk-train.conll')
    print(dataset.train_data[0])
    print(dataset.train_data.sentence_str[0])