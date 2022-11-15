import json
from typing import List


def parse_label(
    label: str,
    use_comb_token: bool,
    unk_token: str = "[UNK]",
    comb_token: str = "[COMB]",
    ignore_tokens: List[str] = ["[MASK]"],
) -> str:
    if label in ignore_tokens:
        return label

    # Normalize the glyph label
    RM_STRS = ["=", "None"]
    for c in RM_STRS:
        label = label.replace(c, "")

    # Replace brackets
    for c in ["（", "〈", "["]:
        label = label.replace(c, "(")
    for c in ["）", "〉", "]"]:
        label = label.replace(c, ")")

    if label == "":
        return unk_token

    if label[-1] == ")":
        for i in range(len(label) - 2, -1, -1):
            if label[i] == "(":
                # "（*）"
                if label[i] == "(":
                    if label[i + 1 : -1] == "○":
                        label = label[:i]
                    else:
                        label = label[i + 1 : -1]
                else:
                    # "*}（*）"
                    if label[i - 1] == "}":
                        label = label[i + 1 : -1]
                    # "A（*）" -> "A"
                    else:
                        label = label[0]
                break
        else:
            label = label[:-1]
    # "A→B"
    if "→" in label:
        label = label.split("→")[1]
    if label == "𬨭":
        label = "將"
    if label == "𫵖":
        label = "尸示"

    if use_comb_token:
        if len(label) != 1:
            return comb_token

    DISCARD_CHARS = ["?" "□", "■", "○", "●", "△", "▲", "☆", "★", "◇", "◆", "□"]
    if any(c in label for c in DISCARD_CHARS):
        return unk_token
    return label


def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def dump_json(obj, path):
    json.dump(
        obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
