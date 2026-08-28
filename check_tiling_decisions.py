from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_page_dataframe(
    manifest: dict,
    image_dir: Path,
) -> pd.DataFrame:
    rows = []

    for page_stem, page in manifest.items():
        # Only process original page entries.
        # Tile entries also have "source", so distinguish them
        # by checking whether "tiles" exists.
        if "source" not in page or "tiles" not in page:
            continue

        source = page["source"]
        image_path = image_dir / source

        if not image_path.exists():
            print(f"[WARNING] Image not found: {image_path}")
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        n_tiles = len(page["tiles"])
        aspect_ratio = width / height if height > 0 else 0

        rows.append(
            {
                "page": page_stem,
                "source": source,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "n_tiles": n_tiles,
            }
        )

    return pd.DataFrame(rows)


def inspect_tile_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = (
        df["n_tiles"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts / len(df) * 100
    ).round(2)

    summary = pd.DataFrame(
        {
            "pages": counts,
            "percentage": percentages,
        }
    )

    print("\n" + "=" * 60)
    print("NUMBER OF TILES PER PAGE")
    print("=" * 60)
    print(summary.to_string())

    print("\nHow many pages use >= N tiles?")

    for threshold in [2, 3, 4, 5, 6, 8]:
        n_pages = (df["n_tiles"] >= threshold).sum()
        percentage = n_pages / len(df) * 100

        print(
            f"  >= {threshold} tiles: "
            f"{n_pages:5d} pages "
            f"({percentage:6.2f}%)"
        )

    # Plot
    ax = counts.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_xlabel("Number of tiles")
    ax.set_ylabel("Number of pages")
    ax.set_title("Distribution of Number of Tiles per Page")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir / "tile_count_distribution.png",
        dpi=200,
    )
    plt.close()


def inspect_aspect_ratio(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 60)
    print("ASPECT RATIO")
    print("=" * 60)

    print(
        df["aspect_ratio"]
        .describe()
        .to_string()
    )

    # Histogram
    plt.figure(figsize=(8, 5))

    plt.hist(
        df["aspect_ratio"],
        bins=30,
    )

    plt.xlabel("Aspect ratio (width / height)")
    plt.ylabel("Number of pages")
    plt.title("Distribution of Page Aspect Ratios")

    plt.tight_layout()
    plt.savefig(
        output_dir / "aspect_ratio_distribution.png",
        dpi=200,
    )
    plt.close()


def inspect_aspect_ratio_vs_tiles(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 60)
    print("ASPECT RATIO VS NUMBER OF TILES")
    print("=" * 60)

    # Scatter plot
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["aspect_ratio"],
        df["n_tiles"],
        alpha=0.5,
    )

    plt.xlabel("Aspect ratio (width / height)")
    plt.ylabel("Number of tiles")
    plt.title("Aspect Ratio vs Number of Tiles")

    plt.tight_layout()
    plt.savefig(
        output_dir / "aspect_ratio_vs_tiles.png",
        dpi=200,
    )
    plt.close()

    # Bin aspect ratios
    bins = [
        0,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        float("inf"),
    ]

    labels = [
        "<0.5",
        "0.5-0.75",
        "0.75-1.0",
        "1.0-1.25",
        "1.25-1.5",
        "1.5-2.0",
        "2.0-2.5",
        "2.5-3.0",
        "3.0-4.0",
        ">=4.0",
    ]

    df = df.copy()

    df["aspect_ratio_bin"] = pd.cut(
        df["aspect_ratio"],
        bins=bins,
        labels=labels,
        right=False,
    )

    stats = (
        df.groupby(
            "aspect_ratio_bin",
            observed=True,
        )
        .agg(
            pages=("page", "count"),
            avg_tiles=("n_tiles", "mean"),
            median_tiles=("n_tiles", "median"),
            max_tiles=("n_tiles", "max"),
        )
    )

    stats["avg_tiles"] = stats["avg_tiles"].round(2)

    print("\nAspect ratio -> tile statistics:")
    print(stats.to_string())

    stats.to_csv(
        output_dir / "aspect_ratio_tile_statistics.csv"
    )


def inspect_four_tile_pages(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    four_tiles = df[df["n_tiles"] >= 4].copy()

    print("\n" + "=" * 60)
    print("PAGES WITH >= 4 TILES")
    print("=" * 60)

    print(f"Number of pages: {len(four_tiles)}")

    if len(four_tiles) > 0:
        print("\nStatistics:")

        print(
            four_tiles[
                [
                    "aspect_ratio",
                    "n_tiles",
                    "width",
                    "height",
                ]
            ]
            .describe()
            .to_string()
        )

        four_tiles = four_tiles.sort_values(
            ["n_tiles", "aspect_ratio"],
            ascending=[False, False],
        )

        four_tiles.to_csv(
            output_dir / "pages_with_4_or_more_tiles.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tiling distribution from manifest.json"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json",
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing original images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tiling_inspection"),
        help="Directory for inspection outputs",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Manifest:   {args.manifest}")
    print(f"Image dir:  {args.image_dir}")
    print(f"Output dir: {args.output_dir}")

    # --------------------------------------------------
    # Load manifest
    # --------------------------------------------------

    manifest = load_manifest(args.manifest)

    # --------------------------------------------------
    # Build dataframe
    # --------------------------------------------------

    df = build_page_dataframe(
        manifest=manifest,
        image_dir=args.image_dir,
    )

    # print(df.head())

    # if df.empty:
    #     print(
    #         "\n[ERROR] No valid page entries found."
    #     )
    #     return

    # print(
    #     f"\nLoaded {len(df)} pages."
    # )

    # # Save raw statistics
    # df.to_csv(
    #     args.output_dir / "page_tiling_stats.csv",
    #     index=False,
    # )

    # # --------------------------------------------------
    # # Inspect
    # # --------------------------------------------------

    # inspect_tile_distribution(
    #     df,
    #     args.output_dir,
    # )

    # inspect_aspect_ratio(
    #     df,
    #     args.output_dir,
    # )

    inspect_aspect_ratio_vs_tiles(
        df,
        args.output_dir,
    )

    # inspect_four_tile_pages(
    #     df,
    #     args.output_dir,
    # )

    # print("\n" + "=" * 60)
    # print("DONE")
    # print("=" * 60)

    # print(
    #     f"Results saved to: {args.output_dir}"
    # )


if __name__ == "__main__":
    main()