import argparse
import os
import time

import pandas as pd
import torch

from utils.reader import CoNLLReader, CoNLLUntokenizedReader

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
resume_iob = {'M-RACE': 0, 'B-PRO': 1, 'S-ORG': 2, 'B-LOC': 3, 'B-CONT': 4, 'M-CONT': 5, 'E-LOC': 6, 'M-PRO': 7, 'M-LOC': 8, 'M-TITLE': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12,
              'E-RACE': 13, 'B-EDU': 14, 'S-NAME': 15, 'B-TITLE': 16, 'S-RACE': 17, 'B-NAME': 18, 'B-RACE': 19, 'E-NAME': 20, 'O': 21, 'E-CONT': 22, 'M-EDU': 23, 'E-TITLE': 24, 'E-EDU': 25,
              'M-NAME': 26, 'E-PRO': 27}
weibo_iob = {'O': 0, 'B-PER.NOM': 1, 'E-PER.NOM': 2, 'B-LOC.NAM': 3, 'E-LOC.NAM': 4, 'B-PER.NAM': 5, 'M-PER.NAM': 6, 'E-PER.NAM': 7, 'S-PER.NOM': 8, 'B-GPE.NAM': 9, 'E-GPE.NAM': 10,
             'B-ORG.NAM': 11, 'M-ORG.NAM': 12, 'E-ORG.NAM': 13, 'M-PER.NOM': 14, 'S-GPE.NAM': 15, 'B-ORG.NOM': 16, 'E-ORG.NOM': 17, 'M-LOC.NAM': 18, 'M-ORG.NOM': 19, 'B-LOC.NOM': 20,
             'M-LOC.NOM': 21, 'E-LOC.NOM': 22, 'B-GPE.NOM': 23, 'E-GPE.NOM': 24, 'M-GPE.NAM': 25, 'S-PER.NAM': 26, 'S-LOC.NOM': 27}
msra_iob = {'O': 0, 'S-NS': 1, 'B-NS': 2, 'E-NS': 3, 'B-NT': 4, 'M-NT': 5, 'E-NT': 6, 'M-NS': 7, 'B-NR': 8, 'M-NR': 9, 'E-NR': 10, 'S-NR': 11, 'S-NT': 12}
ontonotes_iob = {'E-PER': 0, 'E-GPE': 1, 'E-LOC': 2, 'M-ORG': 3, 'E-ORG': 4, 'S-ORG': 5, 'B-GPE': 6, 'O': 7, 'M-PER': 8, 'M-LOC': 9, 'B-PER': 10, 'M-GPE': 11, 'S-LOC': 12, 'B-ORG': 13,
                 'S-PER': 14, 'B-LOC': 15, 'S-GPE': 16}


def get_tagset(tagging_scheme):
    if os.path.isfile(tagging_scheme):
        # read the tagging scheme from a file
        sep = '\t' if tagging_scheme.endswith('.tsv') else ','
        df = pd.read_csv(tagging_scheme, sep=sep)
        tags = {row['tag']: row['idx'] for idx, row in df.iterrows()}
        return tags

    if 'conll' in tagging_scheme:
        return conll_iob
    elif 'wnut' in tagging_scheme:
        return wnut_iob
    elif 'resume' in tagging_scheme:
        return resume_iob
    elif 'ontonotes' in tagging_scheme:
        return ontonotes_iob
    elif 'msra' in tagging_scheme:
        return msra_iob
    elif 'weibo' in tagging_scheme:
        return weibo_iob


def get_reader(file_path, max_instances=-1, max_length=50, tagging_scheme='wnut', encoder_model=None):        
    target_vocab = get_tagset(tagging_scheme)

    if encoder_model is None:
        reader = CoNLLUntokenizedReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab)
    else:
        reader = CoNLLReader(max_instances=max_instances, max_length=max_length,
                             target_vocab=target_vocab, encoder_model=encoder_model)
    
    reader.read_data(file_path)
    return reader
