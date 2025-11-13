# MNIST Test Asset Preparation

This helper script converts the MNIST test split from the original IDX format
into lightweight binary blobs that can be fetched directly in the browser.

## Usage

From the project root, run:

```bash
python tools/mnist_assets/prepare_mnist_test_assets.py
```

By default the script reads the raw IDX files from `data/MNIST/raw/` and writes
three artefacts into `assets/data/`:

- `mnist-test-images-uint8.bin` – concatenated image bytes (uint8, row-major, 28×28 each)
- `mnist-test-labels-uint8.bin` – the label for each image (uint8 digits `0`–`9`)
- `mnist-test-manifest.json` – metadata describing the blob layout

All paths and filenames can be overridden with command-line flags (`--images`,
`--labels`, `--output-dir`, `--image-filename`, `--label-filename`,
`--manifest-filename`).

## Output format

The browser loads `mnist-test-manifest.json` to discover the accompanying binary
files. Image pixels remain as raw uint8 values to keep the payload compact
(≈7.8 MB for all 10,000 test samples). The client is responsible for normalising
pixels to `[0, 1]` before feeding them into the network visualisation.
