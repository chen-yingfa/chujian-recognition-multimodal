from pathlib import Path
from argparse import Namespace, ArgumentParser

from data.dataset import ChujianSeqDataset
from trainer import Trainer
from modeling.vit_bert import VitBert


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--mode", type=str, default="train_test")
    p.add_argument("--vocab_path", type=Path, default="./data/vocab_k3.json")
    p.add_argument(
        "--train_path", type=Path, default="../data/sequences/train.json"
    )
    p.add_argument(
        "--dev_path", type=Path, default="../data/sequences/dev.json"
    )
    p.add_argument(
        "--test_path", type=Path, default="../data/sequences/test.json"
    )

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)

    # BERT parameters
    p.add_argument(
        "--bert_config_file",
        type=Path,
        default="./configs/roberta-base-config.json",
    )
    p.add_argument("--context_len", type=int, default=8)

    # Vit parameters
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--vit_ckpt", type=Path, default="../vit_k3.pt")
    return p.parse_args()


def main():
    args = parse_args()

    img_size = (args.img_size, args.img_size)
    NUM_CLASSES = 2476

    model = VitBert(
        num_classes=NUM_CLASSES,
        img_size=img_size,
        patch_size=args.patch_size,
        bert_config_file=args.bert_config_file,
        vit_ckpt=args.vit_ckpt,
    )

    output_dir = Path(
        "result",
        "temp_vit_bert",
        f"lr{args.lr}",
    )

    trainer = Trainer(
        model,
        output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        log_interval=5,
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
        trainer.train(train_data, dev_data)
    if "test" in args.mode:
        test_output_dir = output_dir / 'test'
        test_data = ChujianSeqDataset(
            data_path=args.test_path,
            vocab_path=args.vocab_path,
            context_len=args.context_len,
            is_training=False,
        )
        trainer.evaluate(test_data, test_output_dir)


if __name__ == "__main__":
    main()
