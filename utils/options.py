import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viterbi', default=False, action='store_true')
    parser.add_argument('--lstm', default=False, action='store_true')
    parser.add_argument("--file_path", default='train_dev/en-train.conll', type=str, help='train file_path')
    parser.add_argument("--encoder_model", default='cointegrated/rubert-tiny2', type=str, help='encoder model hf path')
    parser.add_argument("--pretrained_path", default='no', type=str, help='path to pretrained CoBert')
    parser.add_argument("--train_max_pairs", default=35000, type=int, help='max_pairs for contrastive learning')
    parser.add_argument("--val_max_pairs", default=3000, type=int, help='max_pairs for contrastive learning')
    parser.add_argument("--max_instances", default=-1, type=int, help='max_instances for CoNLLDataset')

    opt = parser.parse_args()
    return opt
