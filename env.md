# Environment

| Item | Value |
|---|---|
| Operating system | Windows 11 (x86-64) |
| CPU | Intel Core i7-10875H @ 2.30 GHz |
| RAM | 16GB |
| GPU | NVIDIA GeForce RTX 2060, 6 GB VRAM |
| NVIDIA driver | 610.62 |
| CUDA runtime | 12.1 (bundled with the torch wheel) |
| cuDNN | 9.1.0 (90100) |
| Python | 3.11.18 |
| Conda environment | `resnet-cifar10-replication` |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| numpy | 1.26.4 |
| Git | 2.56.0.windows.4 |
| Determinism | `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| Seeds | 0, 1, 2 |
| Dataset | CIFAR-10, 50 000 train / 10 000 test images |
| Validation split | 45 000 train / 5 000 val (`--val-split 5000 --seed S`) |
| Pretrained weights | fetched by `scripts/download_pretrained.py`, not tracked by git |
