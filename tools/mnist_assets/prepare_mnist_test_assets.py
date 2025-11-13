"""Prepare browser-friendly MNIST test assets for the visualisation.

This script converts the canonical IDX files from the MNIST test split
into compact binary blobs that are easy to fetch in the browser.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path


MNIST_MAGIC_IMAGES = 2051
MNIST_MAGIC_LABELS = 2049
DEFAULT_IMAGE_FILENAME = "mnist-test-images-uint8.bin"
DEFAULT_LABEL_FILENAME = "mnist-test-labels-uint8.bin"
DEFAULT_MANIFEST_FILENAME = "mnist-test-manifest.json"


def read_idx_bytes(path: Path) -> bytes:
    """Return the raw bytes for an IDX file, transparently handling .gz files."""
    if path.exists():
        return path.read_bytes()
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as handle:
            return handle.read()
    raise FileNotFoundError(f"Neither {path} nor {gz_path} could be located.")


def load_idx(path: Path, expected_magic: int) -> tuple[tuple[int, ...], bytes]:
    """Load an IDX file and return (shape, payload_bytes)."""
    raw = read_idx_bytes(path)
    if len(raw) < 8:
        raise ValueError(f"{path} is too short to contain a valid IDX header.")
    magic, = struct.unpack_from(">I", raw, 0)
    if magic != expected_magic:
        raise ValueError(f"{path} has magic {magic}, expected {expected_magic}.")
    num_dimensions = raw[3]
    if num_dimensions <= 0:
        raise ValueError(f"{path} reports non-positive dimension count: {num_dimensions}.")
    offset = 4
    shape = []
    for idx in range(num_dimensions):
        offset += 4
        if offset > len(raw):
            raise ValueError(f"{path} has truncated dimension metadata at index {idx}.")
        size = struct.unpack_from(">I", raw, offset - 4)[0]
        if size <= 0:
            raise ValueError(f"{path} has non-positive size for dimension {idx}: {size}.")
        shape.append(size)
    payload = raw[offset:]
    expected_length = 1
    for dim in shape:
        expected_length *= dim
    if len(payload) != expected_length:
        raise ValueError(
            f"{path} payload length mismatch: expected {expected_length}, found {len(payload)}.",
        )
    return tuple(shape), payload


def ensure_matching_counts(image_shape: tuple[int, ...], label_shape: tuple[int, ...]) -> int:
    """Validate that the number of images matches the number of labels."""
    if len(image_shape) < 1 or len(label_shape) < 1:
        raise ValueError("Image and label shapes must each have at least one dimension.")
    num_images = image_shape[0]
    num_labels = label_shape[0]
    if num_images != num_labels:
        raise ValueError(f"Image count ({num_images}) does not equal label count ({num_labels}).")
    return num_images


def write_assets(
    image_bytes: bytes,
    label_bytes: bytes,
    output_dir: Path,
    image_filename: str,
    label_filename: str,
    manifest_filename: str,
    shape: tuple[int, ...],
) -> None:
    """Persist the prepared binary assets and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / image_filename).write_bytes(image_bytes)
    (output_dir / label_filename).write_bytes(label_bytes)
    rows = shape[1] if len(shape) > 1 else 1
    cols = shape[2] if len(shape) > 2 else 1
    manifest = {
        "version": 1,
        "numSamples": shape[0],
        "imageShape": [rows, cols],
        "image": {
            "file": image_filename,
            "dtype": "uint8",
        },
        "labels": {
            "file": label_filename,
            "dtype": "uint8",
        },
    }
    (output_dir / manifest_filename).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MNIST IDX test files into browser-friendly binary assets.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/MNIST/raw/t10k-images-idx3-ubyte"),
        help="Path to the MNIST test images IDX file (extracts .gz automatically if needed).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/MNIST/raw/t10k-labels-idx1-ubyte"),
        help="Path to the MNIST test labels IDX file (extracts .gz automatically if needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/data"),
        help="Directory for the generated assets.",
    )
    parser.add_argument(
        "--image-filename",
        default=DEFAULT_IMAGE_FILENAME,
        help="Filename for the packed test images blob.",
    )
    parser.add_argument(
        "--label-filename",
        default=DEFAULT_LABEL_FILENAME,
        help="Filename for the packed test labels blob.",
    )
    parser.add_argument(
        "--manifest-filename",
        default=DEFAULT_MANIFEST_FILENAME,
        help="Filename for the manifest JSON.",
    )
    args = parser.parse_args()

    image_shape, image_payload = load_idx(args.images, MNIST_MAGIC_IMAGES)
    label_shape, label_payload = load_idx(args.labels, MNIST_MAGIC_LABELS)
    num_samples = ensure_matching_counts(image_shape, label_shape)
    if num_samples == 0:
        raise RuntimeError("No samples found in the MNIST test split.")
    write_assets(
        image_payload,
        label_payload,
        args.output_dir,
        args.image_filename,
        args.label_filename,
        args.manifest_filename,
        image_shape,
    )
    print(
        f"Wrote MNIST test assets for {num_samples} samples to {args.output_dir.resolve()}.\n"
        f"Image blob: {args.image_filename}\n"
        f"Label blob: {args.label_filename}\n"
        f"Manifest : {args.manifest_filename}",
    )


if __name__ == "__main__":
    main()
