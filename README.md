# Region-aware Frequency Divide-and-Conquer for Shadow Removal

This repository is the **official implementation** of the paper **“Region-aware Frequency Divide-and-Conquer for Shadow Removal”**, published in **Expert Systems with Applications (ESWA), 2026**.

## Pretrained models

The pretrained checkpoints are provided in [`checkpoint`](checkpoint):

| Dataset       | Checkpoint                   |
| ------------- | ---------------------------- |
| ISTD-adjusted | `checkpoint/model_aistd.pth` |
| SRD           | `checkpoint/model_srd.pth`   |

## Environment

The recommended environment is:

```text
Python 3.9
PyTorch 2.1
CUDA 11.8
```

The inference code requires an NVIDIA GPU. The other main Python dependencies are:

```text
torch
torchvision
einops
timm
numpy
opencv-python
scikit-image
tqdm
natsort
```

The repository contains a precompiled NATTEN extension named `libnatten.cpython-39-x86_64-linux-gnu.so`, intended for the environment above on 64-bit Linux. If it is incompatible with your system, rebuild the NATTEN extension before running inference.

## Dataset preparation

`test.py` expects `--input_dir` to point to a test-set root with the following layout:

```text
<dataset_root>/
├── test_A/             # shadow images
├── test_B/             # shadow masks
└── test_C/             # shadow-free ground-truth images
```

Each sample must have the same filename stem in all four directories. Both `.png` and `.jpg` images are supported.

### Generate shadow-boundary masks

The shadow-boundary masks can be generated from the binary shadow masks with [`utils/boundary_mask.py`](utils/boundary_mask.py). The script computes a width-7 morphological boundary using the difference between the dilated and eroded masks.

Before running it, edit the following values in `main()`:

```python
path_dir = '/path/to/AISTD/test/'  # test-set root
mask_dir = 'test_B'                # input shadow-mask directory
```

Also change the output directory in the script from `test_Boundary_box_k7` to the directory name expected by `test.py`:

```python
save_path = os.path.join(path_dir, 'test_Boundary_k7')
```

Then run the script from the repository root:

```bash
python utils/boundary_mask.py
```

The generated masks will be saved under `<dataset_root>/test_Boundary_k7/`. Keep their filenames identical to the corresponding files in `test_A`, `test_B`, and `test_C`.

## Testing

Run all commands from the repository root. Always provide `--input_dir`, because the default value in `test.py` is a machine-specific path.

### SRD

```bash
python test.py \
  --input_dir /path/to/SRD/test \
  --weights checkpoint/model_srd.pth \
  --result_dir results/srd \
  --gpus 0
```

### ISTD-adjusted

```bash
python test.py \
  --input_dir /path/to/ISTD_adjusted/test \
  --weights checkpoint/model_aistd.pth \
  --result_dir results/aistd \
  --gpus 0
```

The restored images are written to the directory specified by `--result_dir`, retaining the input filenames. Images whose height is at least 1300 pixels are processed tile by tile; other dimensions are automatically padded to a multiple of 8 and cropped back to their original size after inference.

### Test arguments

| Argument       | Default                    | Description                                                                    |
| -------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `--input_dir`  | machine-specific SRD path  | Test-set root containing the four folders above                                |
| `--result_dir` | `results/srd`              | Output directory                                                               |
| `--weights`    | `checkpoint/model_srd.pth` | Pretrained checkpoint                                                          |
| `--gpus`       | `0`                        | CUDA device ID exposed through `CUDA_VISIBLE_DEVICES`                          |
| `--embed_dim`  | `64`                       | Base model embedding dimension; keep this at `64` for the supplied checkpoints |

## Citation

If this work is useful in your research, please cite the paper:

```bibtex
@article{WANG2026134197,
title = {Region-aware Frequency Divide-and-Conquer for Shadow Removal},
journal = {Expert Systems with Applications},
pages = {134197},
year = {2026},
issn = {0957-4174},
author = {Yinan Wang and Yan Huang and Yong Xu and Patrick Le Callet},
}
```
