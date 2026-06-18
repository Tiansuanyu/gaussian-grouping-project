"""Generate per-object ID visualizations from one rendered training view.

This is a GPU helper intended to be run manually on the remote server.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


SCENES = ("ramen", "figurines", "teatime")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one training view and save one highlighted image per object ID."
    )
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a non-negative integer")
    if args.iteration <= 0:
        parser.error("--iteration must be a positive integer")
    return args


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def load_classifier_state(torch, classifier_path: Path):
    try:
        return torch.load(
            classifier_path,
            map_location="cuda",
            weights_only=True,
        )
    except TypeError:
        return torch.load(classifier_path, map_location="cuda")


def make_dataset_args(source_path: Path, model_path: Path, num_classes: int):
    return SimpleNamespace(
        sh_degree=3,
        source_path=str(source_path.resolve()),
        model_path=str(model_path),
        images="images",
        resolution=1,
        white_background=False,
        data_device="cuda",
        eval=False,
        n_views=100,
        random_init=False,
        train_split=True,
        object_path="object_mask",
        num_classes=num_classes,
    )


def make_pipeline_args():
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )


def id_color(object_id: int) -> tuple[int, int, int]:
    """Return a deterministic, bright RGB color for an object ID."""
    import colorsys

    hue = (object_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def save_highlighted_images(rendering, predicted_ids, output_dir: Path) -> list[int]:
    import numpy as np
    from PIL import Image, ImageDraw

    rgb = (
        rendering.detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    rgb = (rgb * 255).round().astype(np.uint8)
    ids = predicted_ids.detach().cpu().numpy()

    saved_ids: list[int] = []
    for object_id in sorted(int(value) for value in np.unique(ids) if int(value) != 0):
        mask = ids == object_id
        if not mask.any():
            continue

        highlight = (rgb.astype(np.float32) * 0.22).astype(np.uint8)
        color = np.asarray(id_color(object_id), dtype=np.float32)
        selected = rgb[mask].astype(np.float32)
        highlight[mask] = np.clip(selected * 0.55 + color * 0.45, 0, 255).astype(
            np.uint8
        )

        image = Image.fromarray(highlight, mode="RGB")
        draw = ImageDraw.Draw(image)
        label = f"Object ID: {object_id:03d}"
        label_box = draw.textbbox((0, 0), label)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        padding = 8
        draw.rectangle(
            (0, 0, label_width + padding * 2, label_height + padding * 2),
            fill=(0, 0, 0),
        )
        draw.text((padding, padding), label, fill=id_color(object_id))

        image.save(output_dir / f"id_{object_id:03d}.png")
        saved_ids.append(object_id)

    return saved_ids


def main() -> None:
    args = parse_args()

    # This must happen before importing torch or the CUDA renderer.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch

    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    model_path = PROJECT_ROOT / "output" / "lerf" / args.scene
    source_path = PROJECT_ROOT / "data" / "lerf" / args.scene
    classifier_path = (
        model_path
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "classifier.pth"
    )
    point_cloud_path = classifier_path.parent / "point_cloud.ply"
    output_dir = model_path / "id_visualization"

    require_path(model_path, "Model path")
    require_path(source_path, "Source path")
    require_path(classifier_path, "Classifier checkpoint")
    require_path(point_cloud_path, "Gaussian point cloud")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script in the real Remote-SSH GPU environment."
        )

    state_dict = load_classifier_state(torch, classifier_path)
    if "weight" not in state_dict:
        raise KeyError(f"Classifier checkpoint has no 'weight' tensor: {classifier_path}")
    num_classes, feature_channels = state_dict["weight"].shape[:2]

    dataset = make_dataset_args(source_path, model_path, num_classes)
    pipeline = make_pipeline_args()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )
    train_views = scene.getTrainCameras()
    if not train_views:
        raise RuntimeError(f"No training views found under: {source_path}")
    if feature_channels != gaussians.num_objects:
        raise ValueError(
            "Classifier input channels do not match Gaussian object features: "
            f"{feature_channels} != {gaussians.num_objects}"
        )

    classifier = torch.nn.Conv2d(
        gaussians.num_objects,
        num_classes,
        kernel_size=1,
    ).cuda()
    classifier.load_state_dict(state_dict)
    classifier.eval()

    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        background_color,
        dtype=torch.float32,
        device="cuda",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        result = render(train_views[0], gaussians, pipeline, background)
        rendering = result["render"]
        logits = classifier(result["render_object"])
        predicted_ids = torch.argmax(logits, dim=0)

    saved_ids = save_highlighted_images(rendering, predicted_ids, output_dir)
    print(f"Saved IDs: {saved_ids}")
    print(f"Output folder: {output_dir}")
    if not saved_ids:
        print("No non-zero object IDs were predicted in the first training view.")


if __name__ == "__main__":
    main()
