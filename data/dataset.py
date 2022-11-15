import json
from typing import List, Dict
from pathlib import Path

from torch import nn
from torch.utils.data import Dataset

from .utils import load_json, parse_label


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of size `chunk_size`.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
        do_mask: bool = False,
        mask_prob: float = 0.20,  # Mask 20% of the tokens
    ):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.context_len = context_len
        self.do_mask = do_mask

        self.vocab: List[str] = load_json(self.vocab_path)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.examples = self.get_examples(data_path)

    def get_examples(self, data_path: Path):
        """
        Format of the source file.
        [
            {
                'seq_id': "train-13",
                'seq': [
                    {
                        "idx": 8,
                        "glyph": "{弔口}",
                        "image": "path/to/file.png",
                    },
                    ...
                ],
            },
            ...
        ]
        """
        all_seqs = load_json(data_path)
        examples = []
        for seq in all_seqs:
            seq_id = seq["id"]
            seq = seq["sequence"]

            # Merge glyph labels
            for glyph in seq:
                glyph["glyph"] = parse_label(
                    glyph["glyph"], use_comb_token=False
                )

            chunks = chunk_2d_list([seq], self.context_len)
            for chunk in chunks:
                examples.append(
                    {
                        "seq_id": seq_id,
                        "seq": chunk,
                    }
                )
        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        if self.do_mask:
            # Mask 15% 
        else:
            return self.data[idx]
