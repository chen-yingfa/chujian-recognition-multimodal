from pathlib import Path
from argparse import Namespace, ArgumentParser

from data.dataset import ChujianSeqDataset
from trainer import Trainer
from modeling.model import VitBert


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

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)

    # BERT parameters
    p.add_argument(
        "--bert_config_file",
        type=Path,
        default="./configs/robert-base-config.json",
    )
    p.add_argument("--context_len", type=int, default=8)

    # Vit parameters
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    img_size = (args.img_size, args.img_size)
    NUM_CLASSES = 2476

    model = VitBert(
        img_size,
        args.patch_size,
        args.bert_config_file,
        num_classes=NUM_CLASSES,
    )

    trainer = Trainer(
        batch_size=args.batch_size,
    )

    if "train" in args.mode:
        train_data = ChujianSeqDataset(
            data_path=args.train_path,
            vocab_path=args.vocab_path,
            context_len=args.context_len,
            is_training=True,
        )
        dev_data = ChujianSeqDataset(
            data_path=args.dev_path,
            vocab_path=args.vocab_path,
            context_len=args.context_len,
            is_training=False,
        )


if __name__ == "__main__":
    main()
