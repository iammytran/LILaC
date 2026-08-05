from huggingface_hub import snapshot_download

# Thay thế tên mô hình bạn muốn tải vào đây
repo_id = "nvidia/NV-Embed-v1" 

# Tải toàn bộ thư mục mô hình về thư mục 'my_local_model'
snapshot_download(
    repo_id=repo_id, 
    local_dir="./models/NV-Embed-v1", 
    local_dir_use_symlinks=False
)
print("Tải mô hình thành công!")

# get number of images
# from pathlib import Path

# path = Path('/workspace/LILaC/datasets/InfoVQA/parsed_documents/dev')

# # Top-level files only
# file_count = sum(1 for item in path.iterdir() if item.is_file())
# images = [item.stem for item in path.iterdir() if item.is_file()]

# # Count files recursively
# # file_count = sum(1 for item in path.rglob('*') if item.is_file())

# print(f"Number of images: {file_count}")
# print(f"images: {images[:5]}")

# # # get number of folders used doc_layout
# # from pathlib import Path

# # folder_path = Path('/workspace/LILaC/artifacts/InfoVQA/layout/dev')
# # folder_count = sum(1 for item in folder_path.iterdir() if item.is_dir())
# # doc_layout_folders = [item.name for item in folder_path.iterdir() if item.is_dir()]
# # print(f"folder_count: {folder_count}")
# # print(f"doc_layout_folders: {doc_layout_folders[:5]}")

# # # print(f"Number of folders: {folder_count}")

# # # missing_names = list(set(images) - set(doc_layout_folders))
# # doc_set = set(doc_layout_folders)  # Using a set here makes lookup instant (O(1))
# # missing_names = [name for name in images if name not in doc_set]
# # print(f"missing names: {missing_names}")


# # # delete all folders in mineru_outputs except folders in missing_names
# # # import shutil

# # # # Thư mục chứa mineru_outputs
# # mineru_dir = Path('/workspace/LILaC/artifacts/InfoVQA/mineru_outputs/dev') # <-- Nhập chính xác đường dẫn của bạn ở đây

# # # Chuyển missing_names thành set để tìm kiếm tức thì O(1)
# # missing_set = set(missing_names)

# # deleted_count = 0
# # kept_count = 0

# # # Lặp qua tất cả các thư mục con trong mineru_outputs
# # for item in mineru_dir.iterdir():
# #     if item.is_dir():
# #         if item.name not in missing_set:
# #             # Nếu tên thư mục KHÔNG nằm trong missing_names -> XÓA
# #             shutil.rmtree(item)
# #             deleted_count += 1
# #             print(f"Deleted: {item.name}")
# #         else:
# #             # Nếu nằm trong missing_names -> GIỮ LẠI
# #             kept_count += 1
# #             print(f"Kept: {item.name}")

# # print(f"\nFinished! Deleted {deleted_count} folders, kept {kept_count} folders.")