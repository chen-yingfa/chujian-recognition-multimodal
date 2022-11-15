"""
Reformat and split the data into train, dev and test, then add unique ID to
each example.
"""
import json
from pathlib import Path
import random


def dump_json(data, path):
    json.dump(
        data, open(path, "w", encoding="utf8"), indent=4, ensure_ascii=False
    )


def load_json(path):
    return json.load(open(path, "r", encoding="utf8"))


def load_examples(path) -> list:
    all_data = load_json(path)
    examples = []
    for i, seq_data in enumerate(all_data):
        examples.append(seq_data["sequence"])
    return examples


def split_data(examples: list[list]):
    """Split into train, dev, test by 18:1:1"""
    split_idx = [
        int(len(examples) * 0.9),
        int(len(examples) * 0.95),
    ]
    random.seed(0)
    random.shuffle(examples)
    train = examples[: split_idx[0]]
    dev = examples[split_idx[0]: split_idx[1]]
    test = examples[split_idx[1]:]

    # Add unique id
    def add_ids(examples, prefix: str):
        return [
            {"id": f"{prefix}-{i}", "sequence": seq}
            for i, seq in enumerate(examples)
        ]

    train = add_ids(train, "train")
    dev = add_ids(dev, "dev")
    test = add_ids(test, "test")

    return train, dev, test


def mask_examples(
    examples: list, mask_prob: float, mask_token: str = "[MASK]"
):
    """Mask some glyphs in each test example. Inplace."""
    for ex in examples:
        seq = ex["sequence"]
        mask_cnt = max(1, int(len(seq) * mask_prob))
        mask_idx = random.sample(range(len(seq)), mask_cnt)
        for i in mask_idx:
            seq[i] = mask_token


def main():
    SEQS_PATH = Path("../../data/sequences/all_sequences.json")
    DST_DIR = Path("../../data/sequences")
    DST_DIR.mkdir(exist_ok=True, parents=True)
    MASK_PROB = 0.15

    examples = load_examples(SEQS_PATH)
    train, dev, test = split_data(examples)
    mask_examples(test, MASK_PROB)
    mask_examples(dev, MASK_PROB)
    print("# examples")
    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")
    dump_json(train, DST_DIR / "train.json")
    dump_json(dev, DST_DIR / "dev.json")
    dump_json(test, DST_DIR / "test.json")


if __name__ == "__main__":
    main()
