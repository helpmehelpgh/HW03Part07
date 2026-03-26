from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_from_disk
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with trained ImageNet CNN ONNX model")
    parser.add_argument("--onnx_model", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/data/CPE_487-587/imagenet-1k-arrow")
    return parser.parse_args()


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_class_names(data_dir: str) -> list[str]:
    dataset = load_from_disk(data_dir)

    if hasattr(dataset["train"].features["label"], "names"):
        return list(dataset["train"].features["label"].names)

    max_label = max(dataset["train"]["label"])
    return [str(i) for i in range(int(max_label) + 1)]


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx_model).expanduser().resolve()
    image_path = Path(args.image_path).expanduser().resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    class_names = load_class_names(args.data_dir)
    input_array = preprocess_image(str(image_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: input_array})[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    print(f"Image path: {image_path}")
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class name: {pred_name}")


if __name__ == "__main__":
    main()