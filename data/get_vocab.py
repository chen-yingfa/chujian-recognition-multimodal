import json
from pathlib import Path


def main():
    k = 3
    glyph_to_files_path = Path(f"../../data/glyphs_k-{k}/glyph_to_files.json")
    glyph_to_files = json.load(open(glyph_to_files_path, "r", encoding="utf8"))
    vocab = list(glyph_to_files.keys())
    vocab_file = Path(f"./vocab_k{k}.json")
    json.dump(
        vocab,
        open(vocab_file, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )


main()
