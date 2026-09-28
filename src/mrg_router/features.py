from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class FrozenMultimodalEncoder:
    """Frozen BGE-M3/SigLIP2 encoder used by the paper.

    Ordered reference images are encoded independently, then mean-pooled for
    the main protocol. Text and image features are standardized downstream and
    concatenated into one query representation.
    """

    def __init__(self, bge_path: str | Path, siglip_path: str | Path, device: str = "cuda") -> None:
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install mrg-router[features] for feature extraction") from error
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self.text_model = AutoModel.from_pretrained(bge_path).eval().to(self.device)
        self.image_processor = AutoProcessor.from_pretrained(siglip_path)
        self.image_model = AutoModel.from_pretrained(siglip_path).eval().to(self.device)
        for model in (self.text_model, self.image_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, instructions: Sequence[str], reference_paths: Sequence[Sequence[str | Path]]) -> np.ndarray:
        from PIL import Image

        if len(instructions) != len(reference_paths) or any(not refs for refs in reference_paths):
            raise ValueError("every instruction needs a non-empty ordered reference list")
        tokens = self.tokenizer(
            list(instructions), return_tensors="pt", truncation=True, padding=True, max_length=64
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        hidden = self.text_model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].bool()
        text = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

        pooled_references = []
        for paths in reference_paths:
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            image_inputs = {name: value.to(self.device) for name, value in image_inputs.items()}
            encoded = self.image_model.get_image_features(**image_inputs)
            if not isinstance(encoded, torch.Tensor):
                encoded = encoded.pooler_output
            pooled_references.append(encoded.mean(0))
        fused = torch.cat((text, torch.stack(pooled_references)), dim=1)
        return fused.float().cpu().numpy()


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(0)
    std = np.maximum(train.std(0), 1e-5)
    values = tuple((array - mean) / std for array in (train,) + others)
    return tuple(value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12) for value in values)

