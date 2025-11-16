"""Utility for training a small QuickDraw MLP and exporting weights for the visualization."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from topoloss import LaplacianPyramid, TopoLoss
from quickdraw import QuickDrawDataset


def seed_everything(seed: int = 0) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

seed_everything(0)

BASE_DATASET_SIZE = 60_000  # only for timeline scaling

def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SmallMLP(nn.Module):
    """Simple fully connected network for 28x28 images."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int = 10):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1], bias=True))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


@dataclass
class LayerMetadata:
    layer_index: int
    type: str
    name: str
    activation: str
    weight_shape: tuple[int, int]
    bias_shape: tuple[int]


@dataclass
class LayerSnapshot:
    metadata: LayerMetadata
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass
class TimelineMilestone:
    identifier: str
    threshold_images: int
    label: str
    kind: str
    dataset_multiple: float | None = None


def export_model(output_path: Path, layer_metadata: Sequence[LayerMetadata], timeline: Sequence[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {
        "version": 2,
        "dtype": "float16",
        "weights": {
            "storage": "per_snapshot_files",
            "format": "layer_array_v1",
            "precision": "float16",
        },
        "network": build_network_payload(layer_metadata),
        "timeline": list(timeline),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch["image"].to(device), batch["label"].to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def parse_hidden_dims(raw: Sequence[int]) -> list[int]:
    dims = [int(d) for d in raw if int(d) > 0]
    if not dims:
        raise ValueError("At least one positive hidden dimension is required.")
    return dims


def capture_layer_snapshots(model: SmallMLP, activations: Sequence[str]) -> list[LayerSnapshot]:
    dense_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    snapshots: list[LayerSnapshot] = []
    for idx, (layer, activation) in enumerate(zip(dense_layers, activations, strict=False)):
        metadata = LayerMetadata(
            layer_index=idx,
            type="dense",
            name=f"dense_{idx}",
            activation=activation,
            weight_shape=tuple(int(dim) for dim in layer.weight.shape),
            bias_shape=tuple(int(dim) for dim in layer.bias.shape),
        )
        snapshots.append(LayerSnapshot(metadata=metadata, weight=layer.weight.detach().cpu(), bias=layer.bias.detach().cpu()))
    return snapshots


def build_network_payload(layers: Sequence[LayerMetadata]) -> dict[str, Any]:
    if not layers:
        raise ValueError("Layer metadata must contain at least one dense layer.")
    architecture = [layers[0].weight_shape[1]] + [layer.bias_shape[0] for layer in layers]
    return {
        "architecture": architecture,
        "layers": [
            {
                "layer_index": layer.layer_index,
                "type": layer.type,
                "name": layer.name,
                "activation": layer.activation,
                "weight_shape": list(layer.weight_shape),
                "bias_shape": list(layer.bias_shape),
            }
            for layer in layers
        ],
        "input_dim": layers[0].weight_shape[1],
        "output_dim": layers[-1].bias_shape[0],
        "normalization": {"mean": 0.5, "std": 0.5},
    }


def slugify_identifier(value: str) -> str:
    transformed = re.sub(r"[^a-z0-9]+", "-", value.lower())
    transformed = transformed.strip("-")
    return transformed or "snapshot"


def tensor_to_base64(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().to(torch.float16).numpy()
    little_endian = np.ascontiguousarray(array.astype("<f2", copy=False)).view("<u2")
    data = little_endian.tobytes()
    return base64.b64encode(data).decode("ascii")


def write_snapshot_file(snapshots: Sequence[LayerSnapshot], directory: Path, order: int, identifier: str) -> Path:
    slug = slugify_identifier(identifier)
    filename = f"{order:03d}_{slug}.json"
    path = directory / filename
    layers_payload = []
    for snapshot in snapshots:
        meta = snapshot.metadata
        layers_payload.append({
            "layer_index": meta.layer_index,
            "name": meta.name,
            "activation": meta.activation,
            "weights": {"shape": list(meta.weight_shape), "data": tensor_to_base64(snapshot.weight)},
            "biases": {"shape": list(meta.bias_shape), "data": tensor_to_base64(snapshot.bias)},
        })
    payload = {"version": 1, "dtype": "float16", "layers": layers_payload}
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def to_posix_relative(path: Path, base: Path) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = Path(os.path.relpath(path, base))
    return relative.as_posix()


def build_default_timeline(dataset_size: int) -> list[TimelineMilestone]:
    specs: list[tuple[str, str, str, float, float | None]] = [
        ("initial", "Initial weights", "initial", 0.0, None),
        ("dataset_1x", "1x dataset", "dataset_multiple", 1.0, 1.0),
        ("dataset_5x", "5x dataset", "dataset_multiple", 5.0, 5.0),
    ]
    milestones: list[TimelineMilestone] = []
    last_threshold = -1
    for identifier, label, kind, ratio, dataset_multiple in specs:
        threshold_images = max(1, int(round(dataset_size * ratio))) if ratio > 0 else 0
        if threshold_images <= last_threshold:
            threshold_images = last_threshold + 1
        milestones.append(TimelineMilestone(identifier, threshold_images, label, kind, dataset_multiple))
        last_threshold = threshold_images
    return milestones


def format_snapshot_description(milestone: TimelineMilestone, images_seen: int, batches_seen: int, dataset_size: int) -> str:
    if milestone.kind == "initial":
        return "0 images processed (random initialisation)"
    dataset_passes = images_seen / dataset_size if dataset_size else 0.0
    human_images = f"{images_seen:,}"
    batches = f"{batches_seen:,}"
    if milestone.kind == "dataset_multiple" and milestone.dataset_multiple is not None:
        multiplier = f"{milestone.dataset_multiple:g}× dataset"
        return f"{human_images} images • {multiplier} • {batches} batches"
    return f"{human_images} images processed • {batches} batches • {dataset_passes:.2f}× dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small QuickDraw MLP and export weights.")
    parser.add_argument("--epochs", type=int, default=5, help="Minimum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--export-path", type=Path, default=Path("exports/mlp_weights.json"), help="Export path.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None, help="Force device.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, export random weights.")
    parser.add_argument("--tau", type=float, default=10.0, help="Tau parameter for topological loss.")
    parser.add_argument("--downsample-factor", type=int, default=3, help="Downsampling factor for topo loss.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    print(f"Using device: {device}")

    selected_classes = ["face", "eye", "mouth", "arm", "leg", "hand", "foot", "beach", "mountain", "house"]
    num_classes = len(selected_classes)

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=0),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Add random noise
        transforms.Normalize((0.5,), (0.5,))  # QuickDraw images roughly 0-1
    ])

    train_dataset = QuickDrawDataset(
        split="train",
        image_size=(28, 28),
        custom_class_names=selected_classes,
        cache_dir="/research/mayukh/huggingface_cache"
    )
    train_dataset.transform = transform  # <-- THIS IS THE FIX

    test_dataset = QuickDrawDataset(
        split="test",
        image_size=(28, 28),
        custom_class_names=selected_classes,
        cache_dir="/research/mayukh/huggingface_cache"
    )
    test_dataset.transform = transform  # <-- ALSO FIX

    model = SmallMLP(28*28, hidden_dims, num_classes=num_classes).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=max(args.num_workers, 0), pin_memory=device.type=="cuda")
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=max(args.num_workers, 0), pin_memory=device.type=="cuda")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataset_size = len(train_dataset)
    milestones = build_default_timeline(dataset_size)

    hidden_activations = ["relu"] * len(hidden_dims) + ["linear"]

    args.export_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.export_path.parent / args.export_path.stem
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    export_root = args.export_path.parent.resolve()
    layer_metadata: list[LayerMetadata] = []
    timeline_entries: list[dict[str, Any]] = []

    cumulative_loss = 0.0
    images_seen = 0
    global_step = 0
    milestone_index = 0
    last_eval_accuracy = 0.0
    total_required_images = milestones[-1].threshold_images
    required_epochs = math.ceil(total_required_images / dataset_size) if dataset_size else 0
    target_epochs = max(args.epochs, required_epochs)

    def record_snapshot(milestone: TimelineMilestone) -> None:
        nonlocal last_eval_accuracy, layer_metadata
        accuracy = evaluate(model, test_loader, device)
        last_eval_accuracy = accuracy
        snapshots = capture_layer_snapshots(model, hidden_activations)
        if not layer_metadata:
            layer_metadata = [snapshot.metadata for snapshot in snapshots]
        snapshot_path = write_snapshot_file(snapshots, snapshot_dir, len(timeline_entries), milestone.identifier)
        weights_rel_path = to_posix_relative(snapshot_path, export_root)
        entry: dict[str, Any] = {
            "id": milestone.identifier,
            "order": len(timeline_entries),
            "label": milestone.label,
            "kind": milestone.kind,
            "target_images": milestone.threshold_images,
            "images_seen": images_seen,
            "batches_seen": global_step,
            "dataset_passes": images_seen / dataset_size if dataset_size else 0.0,
            "description": format_snapshot_description(milestone, images_seen, global_step, dataset_size),
            "metrics": {"test_accuracy": accuracy},
            "weights": {"path": weights_rel_path, "dtype": "float16", "format": "layer_array_v1"},
        }
        if milestone.dataset_multiple is not None:
            entry["dataset_multiple"] = milestone.dataset_multiple
        if images_seen > 0:
            entry["metrics"]["avg_training_loss"] = cumulative_loss / images_seen
        timeline_entries.append(entry)
        print(f"[Timeline] Captured '{milestone.label}' at {images_seen:,} images ({global_step:,} batches) – accuracy: {accuracy * 100:.2f}%")

    def advance_milestones() -> None:
        nonlocal milestone_index
        while milestone_index < len(milestones) and images_seen >= milestones[milestone_index].threshold_images:
            record_snapshot(milestones[milestone_index])
            milestone_index += 1

    advance_milestones()

    if not args.skip_train:
        training_complete = milestone_index >= len(milestones)
        topo_loss = TopoLoss(
            losses=[
                LaplacianPyramid.from_layer(model=model, layer=model.net[0],
                                            factor_h=args.downsample_factor, factor_w=args.downsample_factor, scale=args.tau),
                LaplacianPyramid.from_layer(model=model, layer=model.net[2],
                                            factor_h=args.downsample_factor, factor_w=args.downsample_factor, scale=args.tau),
            ]
        )
        for epoch in range(1, target_epochs + 1):
            if training_complete:
                break
            model.train()
            epoch_loss = 0.0
            epoch_images = 0
            for batch in train_loader:
                data, target = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                t_loss = topo_loss.compute(model=model)
                loss = loss + t_loss
                loss.backward()
                optimizer.step()

                batch_size = data.size(0)
                images_seen += batch_size
                global_step += 1
                cumulative_loss += loss.item() * batch_size
                epoch_loss += loss.item() * batch_size
                epoch_images += batch_size

                if global_step % 100 == 0:
                    avg_loss = epoch_loss / epoch_images if epoch_images else 0.0
                    train_accuracy = output.argmax(dim=1).eq(target).sum().item() / batch_size
                    print(f"Epoch {epoch:02d} Batch {global_step:,} - avg loss: {avg_loss:.4f} train_accuracy: {train_accuracy * 100:.2f}% - Topo loss: {t_loss.item():.4f}")

                advance_milestones()
                if milestone_index >= len(milestones):
                    training_complete = True
                    break

            avg_epoch_loss = epoch_loss / epoch_images if epoch_images else 0.0
            if not training_complete:
                last_eval_accuracy = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d} - avg loss: {avg_epoch_loss:.4f} - test accuracy: {last_eval_accuracy * 100:.2f}% - images seen: {images_seen:,}")

    if not timeline_entries:
        record_snapshot(milestones[0])

    if not layer_metadata:
        raise RuntimeError("Layer metadata could not be captured for export.")
    export_model(args.export_path, layer_metadata, timeline_entries)
    print(f"Exported weights to {args.export_path.resolve()}")


if __name__ == "__main__":
    main()
