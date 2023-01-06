import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viterbi', default=False, action='store_true')
    parser.add_argument('--lstm', default=False, action='store_true')
    parser.add_argument("--file_path", default='train_dev/en-train.conll', type=str, help='train file_path')
    parser.add_argument("--encoder_model", default='cointegrated/rubert-tiny2', type=str, help='encoder model hf path')

    opt = parser.parse_args()
    return opt
