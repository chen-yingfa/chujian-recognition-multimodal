from pathlib import Path
from argparse import Namespace, ArgumentParser

from data.dataset import ChujianSeqDataset


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--vocab_path", type=Path, default="./data/vocab.json")
    p.add_argument(
        "--train_path", type=Path, default="../data/sequences/train.json"
    )
    p.add_argument(
        "--dev_path", type=Path, default="../data/sequences/dev.json"
    )
    p.add_argument(
        "--test_path", type=Path, default="../data/sequences/test.json"
    )
    p.add_argument("--context_len", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = ChujianSeqDataset(
        data_path=args.dev_path,
        vocab_path=args.vocab_path,
    )
    print(dataset)


if __name__ == "__main__":
    main()
