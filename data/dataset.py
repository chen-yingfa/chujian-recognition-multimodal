from typing import List, Dict
from pathlib import Path

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image

from .utils import load_json, parse_label


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of size `chunk_size`.
    """
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def chunk_2d_list(lst: List[List], chunk_size: int) -> List[List]:
    """
    Split a 2D list into chunks of size `chunk_size`.

    Example, if `chunk_size` is 2, then
    [
        [0, 1, 2],
        [5, 6, 7],
    ]
    will be split into
    [
        [0, 1],
        [2],
        [5, 6],
        [7],
    ]
    """
    chunks = []
    for seq in lst:
        chunks.extend(chunk_list(seq, chunk_size))
    return chunks


class ChujianSeqDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        vocab_path: Path,
        context_len: int = 8,
        is_training: bool = False,
        mask_prob: float = 0.20,  # Mask 20% of the tokens
        img_size: int = 224,
    ):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.context_len = context_len
        self.is_training = is_training
        self.mask_prob = mask_prob
        self.img_size = img_size

        self.vocab: List[str] = load_json(self.vocab_path)
        # NOTE: Text specific tokens must be appended, else the token IDs
        # will be different from the pretrained ViT model.
        self.vocab += ["[MASK]", "[CLS]", "[SEP]", "[UNK]", "[PAD]"]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.mask_token_id = self.token_to_id["[MASK]"]
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.cls_token_id = self.token_to_id["[CLS]"]
        self.sep_token_id = self.token_to_id["[SEP]"]
        self.examples = self.get_examples(data_path)
        self.transform = self.get_transform()

    def get_transform(self):
        if self.is_training:
            return transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    # Data augmentation
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomAdjustSharpness(sharpness_factor=4),
                    # transforms.RandomInvert(),
                    # transforms.RandomAutocontrast(),
                    transforms.RandomGrayscale(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

    def get_examples(self, data_path: Path) -> List[Dict]:
        """
        Format of the source file.
        [
            {
                "seq_id": "train-13",
                "seq": [
                    {
                        "idx": 0,
                        "glyph": "{弔口}",
                        "image": "path/to/file.png",
                    },
                    ...
                ],
                "masks": [3, 4]
            },
            ...
        ]
        """
        all_seqs = load_json(data_path)
        examples = []
        for seq_data in all_seqs:
            seq_id = seq_data["id"]
            seq = seq_data["sequence"]
            masks = None
            if "masks" in seq_data:
                masks = seq_data["masks"]

            # Merge glyph labels
            for glyph in seq:
                glyph["glyph"] = parse_label(
                    glyph["glyph"], use_comb_token=False
                )

            chunks = chunk_list(seq, self.context_len)
            for chunk in chunks:
                examples.append(
                    {
                        "seq_id": seq_id,
                        "seq": chunk,
                        "masks": masks,
                    }
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def tokenize(self, text: str) -> int:
        return self.token_to_id.get(text, self.unk_token_id)

    def add_special_tokens(self, input_ids: list, labels: list):
        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        labels = [-100] + labels + [-100]
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict:
        chunk = self.examples[idx]
        seq = chunk["seq"]
        glyphs = [glyph["glyph"] for glyph in seq]
        image_paths = [glyph["image"] for glyph in seq]
        # seq_id = chunk["seq_id"]

        # Get mask indices
        if self.is_training:
            seq_len = len(seq)
            mask_cnt = max(1, int(seq_len * self.mask_prob))
            mask_indices = set(
                np.random.choice(seq_len, mask_cnt, replace=False)
            )
        else:
            mask_indices = set(chunk["masks"])

        # Tokenize and mask
        input_ids = [self.tokenize(glyph) for glyph in glyphs]
        labels = [-100] * len(input_ids)
        for i, glyph in enumerate(seq):
            if i in mask_indices:
                labels[i] = input_ids[i]
                input_ids[i] = self.mask_token_id
        input_ids, labels = self.add_special_tokens(input_ids, labels)
        attention_mask = [1] * len(input_ids)

        # Load images
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            img = self.transform(img)
            images.append(img)

        # Pad
        pad_len = self.context_len + 2 - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
        images += [torch.zeros(3, self.img_size, self.img_size)] * pad_len

        return {
            "images": torch.stack(images),  # (n, 3, img_size, img_size)
            "input_ids": torch.tensor(input_ids),  # (n + 2)
            "attention_mask": torch.tensor(attention_mask),  # (n + 2)
            "labels": torch.tensor(labels),  # (n + 2)
        }
