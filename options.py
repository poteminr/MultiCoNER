import argparse


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viterbi', default=False, action='store_true')
    opt = parser.parse_args()
    return opt
