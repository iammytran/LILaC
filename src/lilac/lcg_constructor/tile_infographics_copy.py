#!/usr/bin/env python3
"""Build a tile-based InfoVQA corpus using aspect-aware infographic grid tiling."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import (
    caption_images,
)
from src.models.embedder.mmembed import MMEmbed
from src.models.embedder.sharded_runner import encode_one_corpus
from src.utils.utils import REPO_ROOT

DEFAULT_INPUT = Path(REPO_ROOT) / "datasets/InfoVQA/image_components/test"
DEFAULT_ROOT = Path(REPO_ROOT) / "datasets/InfoVQA/tiles"
DEFAULT_ESTIMATES = Path(REPO_ROOT) / "debug/estimated_components.json"
TILE_PROMPT = (
    "The first image is a tile from an infographic and the second image is "
    "the complete original infographic. Describe only the tile, but use the "
    "complete infographic as context. Transcribe all visible text and explain "
    "the tile's spatial role. Be factual and concise."
)
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_tile_grid(
    width: int,
    height: int,
    max_tiles: int = 4,
    estimated: int | None = None,
    min_subcomponents: int = 4,
    min_aspect_ratio: float = 2.0,
) -> tuple[int, int]:
    """Calculate (rows, cols) layout tailored for infographics."""
    aspect = height / max(width, 1)

    # 1. Trần động: nới trần x2 cho infographic siêu dài/rộng (>= 4:1 hoặc <= 1:4)
    effective_max = max_tiles * 2 if (aspect >= 4.0 or aspect <= 0.25) else max_tiles

    # 2. Số lượng tile yêu cầu dựa trên mật độ thành phần
    if estimated:
        requested = max(1, math.ceil(estimated / max(min_subcomponents, 1)))
    else:
        requested = max_tiles

    # 3. Chia nhánh theo tỷ lệ aspect
    if aspect >= min_aspect_ratio:
        # Infographic dài dọc: ép thành 1 cột
        return min(effective_max, requested), 1
    if aspect <= 1.0 / min_aspect_ratio:
        # Infographic trải ngang: ép thành 1 hàng
        return 1, min(effective_max, requested)

    # Infographic cân đối / gần vuông
    total = min(effective_max, requested)
    if total <= 2:
        return 2, 1
    return 2, 2


def _compute_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Calculate slice start positions ensuring full coverage of length."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def load_estimates(path: Path) -> dict[str, int]:
    """Load and validate estimated component counts."""
    if not path.exists():
        raise FileNotFoundError(f"Missing estimated-components file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {
        str(k): v for k, v in raw.items()
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1
    }


def prepare_tiled_inputs(
    images_dir: Path,
    staging_dir: Path,
    manifest_path: Path,
    *,
    tile_width: int | None = None,
    tile_height: int | None = None,
    overlap: int = 0,
    max_tiles: int = 4,
    estimated_components_path: Path = DEFAULT_ESTIMATES,
) -> dict[str, Any]:
    """Tile infographics according to aspect ratio and component density."""
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    estimates = load_estimates(estimated_components_path)
    staging_dir.mkdir(parents=True, exist_ok=True)
    infographics: list[dict[str, Any]] = []

    for source in sorted(images_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in VALID_IMAGE_EXTS:
            continue

        estimated = next(
            (estimates[k] for k in (str(source), str(source.resolve()), source.name, source.stem) if k in estimates),
            None,
        )

        with Image.open(source) as img:
            image = img.convert("RGB")
            width, height = image.size

            # Tính toán kích thước tile dựa trên lưới tile_grid hoặc kích thước chỉ định
            if tile_width is not None and tile_height is not None:
                cur_w, cur_h = tile_width, tile_height
            else:
                rows, cols = resolve_tile_grid(width, height, max_tiles=max_tiles, estimated=estimated)
                cur_w = tile_width or math.ceil(width / cols)
                cur_h = tile_height or math.ceil(height / rows)

            if overlap >= min(cur_w, cur_h):
                raise ValueError(f"Overlap ({overlap}) must be smaller than tile dimensions ({cur_w}x{cur_h})")

            xs = _compute_positions(width, cur_w, cur_w - overlap)
            ys = _compute_positions(height, cur_h, cur_h - overlap)

            infographic_dir = staging_dir / source.stem
            infographic_dir.mkdir(parents=True, exist_ok=True)
            tiles = []

            for row, y in enumerate(ys):
                for col, x in enumerate(xs):
                    w = min(cur_w, width - x)
                    h = min(cur_h, height - y)
                    tile_name = f"{source.stem}__r{row:03d}_c{col:03d}__x{x}_y{y}_w{w}_h{h}.png"
                    tile_path = infographic_dir / tile_name
                    
                    image.crop((x, y, x + w, y + h)).save(tile_path)
                    tiles.append({
                        "component_id": f"i_1_t{len(tiles):04d}",
                        "filename": str(tile_path),
                        "path": str(tile_path),
                        "row": row, "column": col,
                        "x": x, "y": y, "width": w, "height": h,
                    })

            infographics.append({
                "id": source.stem,
                "estimated_components": estimated,
                "original": {"filename": str(source), "path": str(source), "width": width, "height": height},
                "tiles": tiles,
            })

    manifest = {"version": 1, "infographics": infographics}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def caption_tiles(manifest: dict[str, Any], output_dir: Path, num_gpus: int | None) -> None:
    """Generate sidecar summaries for tiles via Qwen-VL."""
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for info in manifest["infographics"]:
        orig = info["original"]["path"]
        for tile in info["tiles"]:
            out_file = summaries_dir / f"{Path(tile['filename']).stem}.txt"
            jobs.append((tile, orig, out_file))

    if not jobs:
        return

    caption_images(
        image_pairs=[[tile["path"], orig] for tile, orig, _ in jobs],
        output_filepaths=[str(out) for _, _, out in jobs],
        prompt=TILE_PROMPT,
        max_tokens=1024,
        num_gpus=num_gpus,
    )

    # Đọc trực tiếp caption vào object manifest
    for tile, _, out in jobs:
        tile["caption"] = out.read_text(encoding="utf-8").strip() if out.exists() else ""

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def serialize_tiles(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Format top-level infographics and tile representations for retriever embedding."""
    serialization_dir = output_dir / "serializations"
    serialization_dir.mkdir(parents=True, exist_ok=True)
    top, low = [], []

    for info in manifest["infographics"]:
        orig_name = Path(info["original"]["filename"]).name
        orig_path = info["original"]["path"]
        top.append({
            "id": [orig_name, "i_1"],
            "target": {
                "text": f"{Path(orig_name).stem} [SEP] original infographic",
                "images": [orig_path],
            },
        })
        for tile in info["tiles"]:
            loc = f"row={tile['row']} column={tile['column']} x={tile['x']} y={tile['y']} width={tile['width']} height={tile['height']}"
            low.append({
                "id": [orig_name, tile["component_id"]],
                "target": {
                    "text": f"{Path(orig_name).stem} [SEP] a tile at location: {loc} [SEP] {tile.get('caption', '')}",
                    "images": [tile["path"]],
                },
            })

    top_path = serialization_dir / "image.json"
    low_path = serialization_dir / "subimage.json"
    top_path.write_text(json.dumps(top, indent=2), encoding="utf-8")
    low_path.write_text(json.dumps(low, indent=2), encoding="utf-8")
    return top_path, low_path


def embed_serializations(top_path: Path, low_path: Path, output_dir: Path, num_gpus: int) -> None:
    """Encode serialized inputs using MM-Embed."""
    embedding_dir = output_dir / "embeddings" / "MM-Embed"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for path, name, bsize in ((top_path, "image", 1), (low_path, "subimage", 4)):
        encode_one_corpus(
            embedder_cls=MMEmbed,
            tmp_prefix=f"tile_{name}_",
            corpus_in_filepath=str(path),
            out_embeddings_path=str(embedding_dir / f"{name}.pt"),
            out_index_path=str(embedding_dir / f"{name}.json"),
            num_gpus=num_gpus,
            max_length=4096,
            batch_size=bsize,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiled InfoVQA Corpus Builder")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--tile-height", type=int, default=None)
    parser.add_argument("--estimated-components", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    manifest = prepare_tiled_inputs(
        args.input_dir,
        args.output_dir,
        args.output_dir / "manifest.json",
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap=args.overlap,
        max_tiles=args.max_tiles,
        estimated_components_path=args.estimated_components,
    )

    if not args.skip_qwen:
        caption_tiles(manifest, args.output_dir, args.num_gpus)

    artifacts_folder = Path("artifacts/InfoVQA")
    top_path, low_path = serialize_tiles(manifest, artifacts_folder)

    embedding_folder = artifacts_folder / "embeddings"
    if not args.skip_embed:
        embed_serializations(top_path, low_path, embedding_folder, args.num_gpus)


if __name__ == "__main__":
    main()