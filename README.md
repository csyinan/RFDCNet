# Region-aware Frequency Divide-and-Conquer for Shadow Removal

This repository is the **official implementation** of the paper **“Region-aware Frequency Divide-and-Conquer for Shadow Removal”**, published in **Expert Systems with Applications (ESWA), 2026**.


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

Install the PyTorch and NATTEN builds matching the recommended environment:

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install "natten==0.15.1+torch210cu118" \
  -f https://whl.natten.org/old/
```



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

The generated masks will be saved under `<dataset_root>/test_Boundary_k7/`.

## Testing


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
