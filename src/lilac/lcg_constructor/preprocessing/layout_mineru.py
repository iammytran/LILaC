#!/usr/bin/env python3
"""
MinerU layout analysis (preprocessing).

Thin wrapper around the upstream MinerU CLI (``mineru`` command). For every
input file (PDF or image) under ``--input_dir`` it produces a structured
``content_list.json`` plus extracted figures/tables under
``<output_dir>/<basename>/``.

The MinerU package and its model weights are installed by
``models/download_layout_analyzers.sh``; you can also run
``mineru-models-download`` manually. Make sure the ``mineru`` executable is
on ``$PATH`` before invoking this script.

Why a wrapper instead of direct API calls: MinerU's API is still evolving
across versions, but the CLI interface (``mineru -p <input> -o <out>``) is
stable. Calling the CLI keeps us decoupled from MinerU internals; if you
need finer-grained control, drop into ``mineru.api`` directly.

Output layout under ``<output_dir>/<basename>/`` (per MinerU defaults):

    <basename>_content_list.json     # ordered list of blocks (text/img/table)
    <basename>_model.json            # raw model output
    <basename>_layout.json           # layout boxes
    images/<...>.jpg                 # extracted figures/tables
    <basename>.md                    # rendered Markdown (optional)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import List


SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="MinerU layout analysis (CLI wrapper).")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory of raw images or PDFs.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write per-document MinerU outputs.")
    parser.add_argument("--backend", default="pipeline",
                        choices=["pipeline", "vlm-transformers", "vlm-vllm-engine", "vlm-sglang-engine"],
                        help="MinerU backend; 'pipeline' is the default modular pipeline.")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "txt", "ocr"],
                        help="MinerU parsing method (pipeline backend only).")
    parser.add_argument("--lang", default=None,
                        help="OCR language hint, e.g. 'en', 'korean'. Optional.")
    parser.add_argument("--extra", default="",
                        help="Extra args forwarded verbatim to the mineru CLI.")
    return parser.parse_args()


def _list_inputs(input_dir: str) -> List[str]:
    files = []
    for entry in sorted(os.listdir(input_dir)):
        if os.path.splitext(entry)[1].lower() in SUPPORTED_EXTS:
            files.append(entry)
    return files


def _ensure_mineru_available():
    if shutil.which("mineru") is None:
        raise FileNotFoundError(
            "The `mineru` CLI was not found on PATH.\n"
            "Install via `pip install mineru[core]` or run "
            "`./models/download_layout_analyzers.sh --only mineru` first."
        )


def _has_content_list(doc_out_dir: str) -> bool:
    """Return True iff doc_out_dir holds any *_content_list.json (resumability marker)."""
    if not os.path.isdir(doc_out_dir):
        return False
    for _root, _dirs, names in os.walk(doc_out_dir):
        for n in names:
            if n.endswith("_content_list.json"):
                return True
    return False


def main():
    args = parse_arguments()
    _ensure_mineru_available()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory not found: {args.input_dir}")
    files = _list_inputs(args.input_dir)
    if not files:
        raise ValueError(f"No supported input files in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resumability: skip docs whose output already has a content_list.json ──
    todo = [f for f in files if not _has_content_list(
        os.path.join(args.output_dir, os.path.splitext(f)[0])
    )]
    print(f"[layout/mineru] {len(todo)}/{len(files)} docs to process (resumable skip)")
    if not todo:
        return

    # ── Single directory-mode invocation (F-019 permanent fix) ───────────────
    # Per-doc `mineru -p <file>` mode pays ~10s model-init overhead PER doc.
    # Staging the remaining docs into a temp dir and invoking `mineru -p <dir>`
    # once amortizes the load across the whole batch (~6× wall-clock speedup
    # for 741-doc MP-DocVQA).
    import tempfile
    cmd_base = ["mineru", "-b", args.backend]
    if args.backend == "pipeline":
        cmd_base += ["-m", args.method]
    if args.lang:
        cmd_base += ["-l", args.lang]
    if args.extra:
        cmd_base += args.extra.split()

    with tempfile.TemporaryDirectory(prefix="mineru_staged_") as staged_in:
        for fname in todo:
            src = os.path.abspath(os.path.join(args.input_dir, fname))
            dst = os.path.join(staged_in, fname)
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass

        cmd = cmd_base + ["-p", staged_in, "-o", args.output_dir]
        print(f"[layout/mineru] {' '.join(cmd[:6])} ... ({len(todo)} staged inputs)")
        # MinerU may emit per-doc errors for tricky pages (e.g. hybrid-auto-engine
        # tensor-dim crashes), but exits 0 for the batch. Don't `check=True` so
        # partial successes are kept; the caller can re-run for missed docs.
        subprocess.run(cmd, check=False)

    # Final summary
    final_done = sum(1 for f in files
                     if _has_content_list(os.path.join(args.output_dir, os.path.splitext(f)[0])))
    print(f"[layout/mineru] after run: {final_done}/{len(files)} docs have content_list.json")


if __name__ == "__main__":
    main()
