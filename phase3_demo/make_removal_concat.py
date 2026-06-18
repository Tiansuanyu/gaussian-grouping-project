"""Build fixed-height comparison images for object-removal renders.

This is a CPU-only post-processing helper. It does not import or run any
Gaussian rendering, training, or CUDA code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


SCENES = ("ramen", "figurines", "teatime")
PANEL_FOLDERS = (
    "gt",
    "renders",
    "gt_objects_color",
    "objects_pred",
    "objects_feature16",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESAMPLING = getattr(Image, "Resampling", Image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild object-removal comparison images at a common height."
    )
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    if args.iteration <= 0:
        parser.error("--iteration must be a positive integer")
    return args


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    target_width = max(1, round(image.width * target_height / image.height))
    return image.resize((target_width, target_height), RESAMPLING.LANCZOS)


def find_matching_image(folder: Path, render_file: Path) -> Path | None:
    exact_path = folder / render_file.name
    if exact_path.is_file():
        return exact_path

    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = folder / f"{render_file.stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def load_panel(path: Path, target_height: int) -> Image.Image:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return resize_to_height(rgb_image, target_height).copy()


def concatenate_images(images: list[Image.Image], target_height: int) -> Image.Image:
    total_width = sum(image.width for image in images)
    comparison = Image.new("RGB", (total_width, target_height))
    x_offset = 0
    for image in images:
        comparison.paste(image, (x_offset, 0))
        x_offset += image.width
    return comparison


def main() -> None:
    args = parse_args()
    result_root = (
        PROJECT_ROOT
        / "output"
        / "lerf"
        / args.scene
        / "train"
        / "ours_object_removal"
        / f"iteration_{args.iteration}"
    )
    renders_dir = result_root / "renders"
    output_dir = result_root / "concat_fixed"

    if not result_root.is_dir():
        raise FileNotFoundError(f"Removal result folder does not exist: {result_root}")
    if not renders_dir.is_dir():
        raise FileNotFoundError(
            f"Required reference folder does not exist: {renders_dir}"
        )

    available_folders: list[tuple[str, Path]] = []
    for folder_name in PANEL_FOLDERS:
        folder = result_root / folder_name
        if folder.is_dir():
            available_folders.append((folder_name, folder))
        else:
            warn(f"missing folder, skipping panel: {folder}")

    render_files = sorted(
        path
        for path in renders_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not render_files:
        raise FileNotFoundError(f"No render images found in: {renders_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for render_file in render_files:
        try:
            with Image.open(render_file) as render_image:
                target_height = render_image.height
        except (OSError, ValueError) as exc:
            warn(f"could not read reference render {render_file}: {exc}; skipping frame")
            continue

        panels: list[Image.Image] = []
        used_folders: list[str] = []
        for folder_name, folder in available_folders:
            image_path = find_matching_image(folder, render_file)
            if image_path is None:
                warn(
                    f"frame {render_file.stem} missing in {folder_name}; "
                    "skipping this panel"
                )
                continue
            try:
                panels.append(load_panel(image_path, target_height))
                used_folders.append(folder_name)
            except (OSError, ValueError) as exc:
                warn(f"could not read {image_path}: {exc}; skipping this panel")

        if not panels:
            warn(f"frame {render_file.name} has no readable panels; skipping frame")
            continue

        comparison = concatenate_images(panels, target_height)
        output_path = output_dir / f"{render_file.stem}.png"
        comparison.save(output_path)
        saved_files.append(output_path)
        print(
            f"Saved {output_path.name} "
            f"({comparison.width}x{comparison.height}) from {', '.join(used_folders)}"
        )

    print(f"Saved {len(saved_files)} comparison image(s).")
    print(f"Output folder: {output_dir}")
    if not saved_files:
        raise RuntimeError("No comparison images were created.")


if __name__ == "__main__":
    main()
