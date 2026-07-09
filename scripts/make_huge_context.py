"""Generate deterministic large context files for local huge-context verification.

The script intentionally writes generated artifacts outside source-controlled
fixtures by default, so huge-context examples can be exercised without adding
large files to the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_payload(blocks: int, lines_per_block: int) -> str:
    sections: list[str] = []
    for block_index in range(blocks):
        lines = [f"# Context block {block_index:05d}"]
        for line_index in range(lines_per_block):
            lines.append(
                " ".join(
                    [
                        f"block={block_index:05d}",
                        f"line={line_index:03d}",
                        "topic=context-engineering",
                        "signal=deterministic-fixture",
                        "instruction=preserve-important-head-and-tail-details",
                    ]
                )
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic large context text file.")
    parser.add_argument("--output", default="artifacts/context/huge_context_sample.txt", help="Output file path.")
    parser.add_argument("--blocks", type=int, default=200, help="Number of repeated context blocks.")
    parser.add_argument("--lines-per-block", type=int, default=12, help="Lines written inside each block.")
    args = parser.parse_args()

    if args.blocks <= 0 or args.lines_per_block <= 0:
        raise SystemExit("--blocks and --lines-per-block must be positive integers")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args.blocks, args.lines_per_block)
    output.write_text(payload, encoding="utf-8")
    approx_tokens = max(1, len(payload) // 4)
    print(f"wrote {output} ({len(payload):,} bytes, ~{approx_tokens:,} heuristic tokens)")


if __name__ == "__main__":
    main()
