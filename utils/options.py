import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viterbi', default=False, action='store_true')
    parser.add_argument("--file_path", default='train_dev/uk-train.conll', type=str, help='train file_path')
    opt = parser.parse_args()
    return opt
