import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _env_or(name, default):
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_path(name, default):
    return Path(_env_or(name, default))


DATA_ROOT = _env_path("SAMDUP_DATA_ROOT", PROJECT_ROOT / "data")
OUTPUT_DIR = _env_path("SAMDUP_OUTPUT_DIR", PROJECT_ROOT / "outputs")
LOG_DIR = _env_path("SAMDUP_LOG_DIR", PROJECT_ROOT / "logs")
SAM_CHECKPOINT = _env_path("SAMDUP_SAM_CHECKPOINT", PROJECT_ROOT / "sam_vit_b_01ec64.pth")

# Script defaults (override via env or CLI args)
DEFAULT_MAIN_IMAGE = _env_or(
    "SAMDUP_MAIN_IMAGE",
    str(DATA_ROOT / "main_image.png"),
)

DEFAULT_COMPARE_IMG1 = _env_or(
    "SAMDUP_COMPARE_IMG1",
    str(DATA_ROOT / "compare_img1.jpg"),
)
DEFAULT_COMPARE_IMG2 = _env_or(
    "SAMDUP_COMPARE_IMG2",
    str(DATA_ROOT / "compare_img2.jpg"),
)
DEFAULT_COMPARE_OUTPUT = _env_or(
    "SAMDUP_COMPARE_OUTPUT",
    str(OUTPUT_DIR / "compare"),
)

DEFAULT_NEW_IDEA_IMG1 = _env_or(
    "SAMDUP_NEW_IDEA_IMG1",
    str(DATA_ROOT / "new_idea_img1.jpg"),
)
DEFAULT_NEW_IDEA_IMG2 = _env_or(
    "SAMDUP_NEW_IDEA_IMG2",
    str(DATA_ROOT / "new_idea_img2.jpg"),
)
DEFAULT_NEW_IDEA_OUTPUT = _env_or(
    "SAMDUP_NEW_IDEA_OUTPUT",
    str(OUTPUT_DIR / "new_idea"),
)

DEFAULT_TEST_IMG1 = _env_or(
    "SAMDUP_TEST_IMG1",
    str(DATA_ROOT / "test_img1.png"),
)
DEFAULT_TEST_IMG2 = _env_or(
    "SAMDUP_TEST_IMG2",
    str(DATA_ROOT / "test_img2.png"),
)

DEFAULT_LOG_FILE = _env_or(
    "SAMDUP_LOG_FILE",
    str(LOG_DIR / "run.log"),
)

DEFAULT_DEBUG_DIR = _env_or(
    "SAMDUP_DEBUG_DIR",
    str(OUTPUT_DIR / "debug"),
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
