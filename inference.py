from utils.options import train_options
from models.baseline_models import Bert, BertCRF, BertBiLstmCRF
from models.siamese_model import CoBertCRF
from utils.dataset import CoNLLDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import compress

def prediction_loop(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    output_str = ''
    for _, (input_ids, _, attention_mask) in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tags = model.predict_tags(input_ids, attention_mask)
        
        for b_input_id, preds in zip(input_ids, tags):
            tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(b_input_id.to('cpu').numpy())

            new_tokens, new_preds = [], []
            for token, pred in zip(tokens, preds):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_preds.append(dataloader.dataset.id_to_label[pred])
                    new_tokens.append(token)
                            
            output_str += '\n'.join(new_preds[1:-1])
            output_str += '\n\n\n'
        
    open('prediction.txt', 'wt').write(output_str) 


if __name__ == '__main__':
    arguments = train_options()
    dataset = CoNLLDataset(
        file_path=arguments.file_path,
        viterbi_algorithm=arguments.viterbi,
        encoder_model=arguments.encoder_model
    )
    
    model = CoBertCRF(
        encoder_model=arguments.encoder_model,
        pretrained_encoder_model_path=arguments.encoder_pretrained_path,
        label_to_id=dataset.label_to_id
    )
    if arguments.model_pretrained_path is not None:
        model.load_state_dict(torch.load(arguments.model_pretrained_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.data_collator, shuffle=False, drop_last=False)
    prediction_loop(model, dataloader)