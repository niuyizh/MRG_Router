from __future__ import annotations

from pathlib import Path
import yaml


class CostRegistry:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path) -> "CostRegistry":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def cost(self, generator_id: str, reference_count: int) -> float:
        row = self.payload["generators"][generator_id]
        if "usd_per_image" in row:
            return float(row["usd_per_image"])
        seconds = float(row["median_seconds_by_reference_count"][reference_count])
        gpu_count = int(row.get("gpu_count", 1))
        rate = float(self.payload["gpu"]["usd_per_gpu_hour"])
        return gpu_count * seconds * rate / 3600.0

