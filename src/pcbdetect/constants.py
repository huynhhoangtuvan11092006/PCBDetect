"""Constants and domain definitions for PCBDetect."""

from pathlib import Path

# Defect class mapping for DeepPCB benchmark
# ID: 1-indexed in DeepPCB annotations
DEFECT_ID_TO_NAME: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",  # Spurious copper
    6: "pin_hole",
}

DEFECT_NAME_TO_ID: dict[str, int] = {v: k for k, v in DEFECT_ID_TO_NAME.items()}

# YOLO zero-indexed class mapping (0 to 5)
YOLO_ID_TO_NAME: dict[int, str] = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin_hole",
}

YOLO_NAME_TO_ID: dict[str, int] = {v: k for k, v in YOLO_ID_TO_NAME.items()}

DEFECT_CLASSES: list[str] = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]

# Distinct vibrant colors for visualization and UI overlays (Hex and RGB)
DEFECT_COLORS: dict[str, str] = {
    "open": "#E63946",  # Vibrant Red
    "short": "#457B9D",  # Steel Blue
    "mousebite": "#F4A261",  # Sandy Orange
    "spur": "#2A9D8F",  # Emerald Green
    "copper": "#9B5DE5",  # Royal Purple
    "pin_hole": "#F15BB5",  # Neon Pink
}

DEFECT_RGB: dict[str, tuple[int, int, int]] = {
    "open": (230, 57, 70),
    "short": (69, 123, 157),
    "mousebite": (244, 162, 97),
    "spur": (42, 157, 143),
    "copper": (155, 93, 229),
    "pin_hole": (241, 91, 181),
}
DEFECT_COLORS_RGB = DEFECT_RGB

# Image specs for DeepPCB
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 640
DEFAULT_CHANNELS = 3

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "DeepPCB" / "PCBData"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
