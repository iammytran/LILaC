from pathlib import Path
tiled_inputs_depth_1_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/tiled_input_depth_1"
outputs_tile_depth_1_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr"

def main():
    count_tiled_inputs_depth_1_folder = len(list(Path(tiled_inputs_depth_1_folder).iterdir()))
    count_outputs_tile_depth_1_folder = len(list(Path(outputs_tile_depth_1_folder).iterdir()))
    print(f"count_tiled_inputs_depth_1_folder: {count_tiled_inputs_depth_1_folder}")
    print(f"count_outputs_tile_depth_1_folder: {count_outputs_tile_depth_1_folder}")
    inputs = []
    outputs = []
    for item in Path(tiled_inputs_depth_1_folder).iterdir():
        if item.stem != "tiling_manifest_depth_1":
            inputs.append(item.stem)

    for item in Path(outputs_tile_depth_1_folder).iterdir():
        outputs.append(item.stem)

    not_process = set(inputs) - set(outputs)
    print(f"not_process: {not_process}")

    # # save results in a file txt
    # output_file = Path("debug/tiles_not_processed.txt")
    # output_file.write_text("\n".join(sorted(not_process)) + "\n", encoding="utf-8")
    return


if __name__ == "__main__":
    main()