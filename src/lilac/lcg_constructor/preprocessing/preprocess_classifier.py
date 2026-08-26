from pathlib import Path
from typing import Literal
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor


class InfographicRouter:
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_name
        ).to(device)

        # Định nghĩa 2 nhóm nhãn semantic trực quan
        self.labels = [
            "a structured document with tables, clean columns, and grid text blocks",  # MinerU
            "a complex graphical infographic with illustrations, non-linear text, and colorful visual elements",  # Qwen-VL
        ]

    def classify_with_siglip(self, image: Image.Image) -> tuple[Literal["mineru", "qwen_vl"], float]:
        """Dùng SigLIP Zero-Shot để phân loại cấu trúc layout."""
        inputs = self.processor(
            text=self.labels,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image) if hasattr(outputs, "logits_per_image") else torch.softmax(logits_per_image, dim=1)
            probs = probs[0].cpu().numpy()

        mineru_score, qwen_score = probs[0], probs[1]
        decision = "qwen_vl" if qwen_score > mineru_score else "mineru"
        confidence = float(max(mineru_score, qwen_score))
        return decision, confidence

    @staticmethod
    def classify_with_heuristic(image_path: Path) -> tuple[Literal["mineru", "qwen_vl"], dict]:
        """
        Phương pháp heuristic OpenCV (nhanh, không dùng GPU):
        - Infographic đồ họa thường có độ phức tạp màu sắc cao (nhiều màu) và nhiều cạnh cong.
        - Tài liệu chuẩn (MinerU) thường có nền đơn sắc, tỷ lệ trắng/đen cao.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return "mineru", {}

        # 1. Đếm số màu độc bản (Color Palette Diversity)
        small_img = cv2.resize(img_bgr, (150, 150))
        unique_colors = len(np.unique(small_img.reshape(-1, 3), axis=0))

        # 2. Phân tích mật độ cạnh (Edge Density)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Ngưỡng heuristic
        # Infographic đồ họa: > 2500 màu độc bản trên thumbnail và mật độ cạnh > 0.05
        is_complex = unique_colors > 2200 and edge_ratio > 0.045
        decision = "qwen_vl" if is_complex else "mineru"

        metrics = {
            "unique_colors": unique_colors,
            "edge_ratio": round(float(edge_ratio), 4),
        }
        return decision, metrics

    def route(self, image_path: str | Path, method: Literal["siglip", "heuristic"] = "siglip") -> str:
        path = Path(image_path)
        if method == "siglip":
            with Image.open(path).convert("RGB") as img:
                choice, conf = self.classify_with_siglip(img)
                print(f"[{path.name}] Model Router: -> {choice.upper()} (Confidence: {conf:.2f})")
                return choice
        else:
            choice, metrics = self.classify_with_heuristic(path)
            print(f"[{path.name}] Heuristic Router: -> {choice.upper()} (Metrics: {metrics})")
            return choice


if __name__ == "__main__":
    router = InfographicRouter()

    # Thử nghiệm với các ảnh
    test_images = [
        "datasets/InfoVQA/image_components/test/10022.jpeg",  #mineru
        "datasets/InfoVQA/image_components/test/10065.jpeg",  #mineru
        "datasets/InfoVQA/image_components/test/70572.jpeg",  #mineru
        "datasets/InfoVQA/image_components/test/43600.jpeg",  #qwen
        "datasets/InfoVQA/image_components/test/43476.jpeg",  #qwen
        "datasets/InfoVQA/image_components/test/10249.jpeg",  #qwen
        "datasets/InfoVQA/image_components/test/44700.jpeg",  #qwen
        "datasets/InfoVQA/image_components/test/41587.jpeg"   #qwen  
    ]

    for img_file in test_images:
        p = Path(img_file)
        if p.exists():
            selected_pipeline = router.route(p, method="siglip")
            print(f"image_file: {img_file}")
            print(f"==> Điều hướng sang: {selected_pipeline}\n")