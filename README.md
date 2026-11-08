# ResNet-CIFAR10 Replication (MIAE01)

Course project: reproducing He et al., *Deep Residual Learning for Image
Recognition* (CVPR 2016, arXiv:1512.03385) on CIFAR-10.

## Reproduction Metadata

- Upstream repository: https://github.com/akamaster/pytorch_resnet_cifar10
- Starting commit: d5489e8995e81e91ce6b1d69dcc98ad579b0b153 (2021-07-20)
- Upstream history imported on 2026-11-08 (`git fetch upstream` + merge)
- Original author: Yerlan Idelbayev — BSD-2-Clause License (see LICENSE)
- Note on repository size: `pretrained_models/*.th` exist only in the inherited
  upstream history. They are untracked at HEAD and ignored via `.gitignore`;
  weights are re-obtained with `scripts/download_pretrained.py`.
- Files modified/added by this group: `git diff --name-status d5489e8..HEAD`

---


# Proper ResNet Implementation for CIFAR10/CIFAR100 in Pytorch
[Torchvision model zoo](https://github.com/pytorch/vision/tree/master/torchvision/models) provides number of implementations of various state-of-the-art architectures, however, most of them are defined and implemented for ImageNet.
Usually it is straightforward to use the provided models on other datasets, but some cases require manual setup.

For instance, very few pytorch repositories with ResNets on CIFAR10 provides the implementation as described in the [original paper](https://arxiv.org/abs/1512.03385). If you just use the torchvision's models on CIFAR10 you'll get the model **that differs in number of layers and parameters**. This is unacceptable if you want to directly compare ResNet-s on CIFAR10 with the original paper.
The purpose of this repo is to provide a valid pytorch implementation of ResNet-s for CIFAR10 as described in the original paper. The following models are provided:

| Name      | # layers | # params| Test err(paper) | Test err(this impl.)|
|-----------|---------:|--------:|:-----------------:|:---------------------:|
|[ResNet20](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th)   |    20    | 0.27M   | 8.75%| **8.27%**|
|[ResNet32](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th)  |    32    | 0.46M   | 7.51%| **7.37%**|
|[ResNet44](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44-014dd654.th)   |    44    | 0.66M   | 7.17%| **6.90%**|
|[ResNet56](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56-4bfd9763.th)   |    56    | 0.85M   | 6.97%| **6.61%**|
|[ResNet110](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th)  |   110    |  1.7M   | 6.43%| **6.32%**|
|[ResNet1202](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202-f3b1deed.th) |  1202    | 19.4M   | 7.93%| **6.18%**|

This implementation matches description of the original paper, with comparable or better test error.

## How to run?
```bash
git clone https://github.com/akamaster/pytorch_resnet_cifar10
cd pytorch_resnet_cifar10
chmod +x run.sh && ./run.sh
```

## Details of training
Our implementation follows the paper in straightforward manner with some caveats: **First**, the training in the paper uses 45k/5k train/validation split on the train data, and selects the best performing model based on the performance on the validation set. We *do not perform* validation testing; if you need to compare your results on ResNet head-to-head to the orginal paper keep this in mind. **Second**, if you want to train ResNet1202 keep in mind that you need 16GB memory on GPU.

## Pretrained models for download
1. [ResNet20, 8.27% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th)
2. [ResNet32, 7.37% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32.th)
3. [ResNet44, 6.90% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44.th)
4. [ResNet56, 6.61% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56.th)
5. [ResNet110, 6.32% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110.th)
6. [ResNet1202, 6.18% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202.th)

If you find this implementation useful and want to cite/mention this page, here is a bibtex citation:

```bibtex
@misc{Idelbayev18a,
  author       = "Yerlan Idelbayev",
  title        = "Proper {ResNet} Implementation for {CIFAR10/CIFAR100} in {PyTorch}",
  howpublished = "\url{https://github.com/akamaster/pytorch_resnet_cifar10}",
  note         = "Accessed: 20xx-xx-xx"
}

```

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11.18 (`conda` environment `resnet-cifar10-replication`) |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| GPU | NVIDIA GeForce RTX 2060, 6 GB, driver 610.62 |

Full record: [env.md](env.md).

## Installation

```bat
git clone https://github.com/yuzhee114/resnet-cifar10-replication.git
cd resnet-cifar10-replication

conda create -n resnet-cifar10-replication python=3.11 -y
conda activate resnet-cifar10-replication

pip install -r requirements.txt
```

## Data

CIFAR-10 is downloaded by `torchvision` into `./data` on the first run and is
ignored by git.

- archive: `data/cifar-10-python.tar.gz`, 170 498 071 bytes
- MD5: `c58f30108f718f92721af3b95e74349a`

## Pretrained weights

The released checkpoints are not tracked by git:

```bat
python scripts/download_pretrained.py
```

## Reproducing Table 6

The paper trains for 64k iterations with batch size 128, which is 391
iterations per epoch and therefore 164 epochs, with the learning rate divided
by 10 at 32k and 48k iterations (epochs 82 and 123).

```bat
python trainer.py --arch=resnet20 --epochs=164 --lr-milestones=82,123 ^
    --seed=0 --val-split=5000 -j 2 --save-dir=save_resnet20_s0 ^
    --log-csv=logs/resnet20_seed0.csv --config-out=logs/resnet20_seed0.json
```

ResNet-110 and ResNet-1202 additionally use the paper's warm-up:

```bat
python trainer.py --arch=resnet110 --epochs=164 --lr-milestones=82,123 ^
    --warmup-iters=400 --seed=0 --val-split=5000 -j 2 ^
    --save-dir=save_resnet110_s0 --log-csv=logs/resnet110_seed0.csv
```

ResNet-1202 (19.4 M parameters) requires about 16 GB of VRAM and is not run on
this 6 GB GPU.

All models, three seeds:

```powershell
foreach ($arch in "resnet20","resnet32","resnet44","resnet56","resnet110") {
  foreach ($seed in 0,1,2) {
    python trainer.py --arch=$arch --epochs=164 --lr-milestones=82,123 `
      --seed=$seed --val-split=5000 -j 2 `
      --save-dir="save_${arch}_s${seed}" `
      --log-csv="logs/${arch}_seed${seed}.csv" `
      --config-out="logs/${arch}_seed${seed}.json"
  }
}
```

## Outputs

Every run prints one line per epoch and ends with a single result line:

```
RESULT arch=resnet20 seed=0 epochs=164 best_val=<val> test_acc=<acc> test_err=<err>
```

`test_err` is the test-set classification error in percent, i.e. the quantity
reported in Table 6 of the paper.

## Runtime

| Model | Params | Test err, paper (Table 6) | Test err, ours (mean ± std, 3 seeds) | Wall-clock, 164 epochs |
|---|---|---|---|---|
| ResNet-20 | 0.27 M | 8.75 | TODO | 75 min (measured: 27.4 s/epoch) |
| ResNet-32 | 0.46 M | 7.51 | TODO | TODO |
| ResNet-44 | 0.66 M | 7.17 | TODO | TODO |
| ResNet-56 | 0.85 M | 6.97 | TODO | TODO |
| ResNet-110 | 1.7 M | 6.43 (6.61 ± 0.16) | TODO | TODO |
| ResNet-1202 | 19.4 M | 7.93 | not reproduced | – |

## Logs

| Path | Content |
|---|---|
| `logs/<arch>_seed<S>.csv` | per-epoch train/val loss and accuracy, learning rate, wall-clock |
| `logs/<arch>_seed<S>.json` | resolved configuration, torch/CUDA versions, GPU name, split sizes |
| `results/summary.csv` | mean ± std per model, aggregated from the per-run CSVs |

Checkpoints (`save_*/`, `*.th`) and `data/` are ignored by git.
