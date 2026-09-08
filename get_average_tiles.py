#!/usr/bin/env python3
"""Calculate the average number of PNG tiles per infographic."""

from pathlib import Path


TILES_DIR = Path("/workspace/LILaC/datasets/InfoVQA/tiles")


def get_tile_counts(tiles_dir: Path) -> dict[str, int]:
    """Return the number of PNG tiles in each infographic directory."""
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    return {
        infographic_dir.name: sum(
            1
            for file_path in infographic_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() == ".png"
        )
        for infographic_dir in sorted(tiles_dir.iterdir())
        if infographic_dir.is_dir()
    }


def main() -> None:
    tile_counts = get_tile_counts(TILES_DIR)
    if not tile_counts:
        print(f"No infographic directories found in {TILES_DIR}")
        return

    total_tiles = sum(tile_counts.values())
    infographic_count = len(tile_counts)
    average_tiles = total_tiles / infographic_count

    print(f"{average_tiles:.2f}")


if __name__ == "__main__":
    main()