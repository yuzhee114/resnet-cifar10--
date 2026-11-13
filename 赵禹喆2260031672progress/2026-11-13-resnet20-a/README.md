# ResNet-20 weekly progress

赵禹喆 · 9–15 November 2026

This week I completed two full ResNet-20 runs on CIFAR-10 with the group’s shared trainer. Both ran for 164 epochs from random initialization. Seed 2 is pending.

## Experimental setup

Both runs used the group trainer at commit f1acc6e, a batch size of 128, SGD with initial learning rate 0.1, momentum 0.9 and weight decay 0.0001. Each seed used 45,000 training images and 5,000 validation images. Training used random flips and padded crops. The trainer selected the checkpoint with the highest validation accuracy and evaluated it on the official 10,000-image test set.

The runs used a macOS arm64 CPU, Python 3.14.7, PyTorch 2.15.0, torchvision 0.29.2, NumPy 2.5.4 and four OpenMP threads. The run-level logs recorded 294.0 minutes for seed 0 and 295.5 minutes for seed 1.

## Results

| Seed | Best epoch | Validation acc. | Test acc. | Test error |
| --- | --- | --- | --- | --- |
| 0 | 139 | 91.740% | 90.870% | 9.130% |
| 1 | 155 | 92.280% | 91.360% | 8.640% |

The preliminary mean test error is 8.89% with a sample standard deviation of 0.35 percentage points across two seeds. The mean is +0.14 percentage points relative to the 8.75% reported in Table 6. One run fell above the paper value and one below. The final course comparison will use all three seeds.

The shared trainer used 57,728 updates, with learning rate drops after 28,864 and 43,296 updates. The paper reports 64,000 updates, with drops after 32,000 and 48,000. The trainer uses fixed channel normalization and the paper describes per-pixel mean subtraction. The current results do not establish how much either difference affects test error.

## Reproduction and evidence

Use the [group trainer at f1acc6e](https://github.com/yuzhee114/resnet-cifar10--replication/commit/f1acc6e872bc3452ef6f2339d57ee1623ec4ff81), based on the [third-party PyTorch implementation](https://github.com/akamaster/pytorch_resnet_cifar10). From the group repository root with CIFAR-10 available, set SEED to 0 or 1 and use fresh output paths.

```sh
SEED=0
RUN="save_runs/a_seed${SEED}"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python trainer.py --arch=resnet20 --epochs=164 --lr-milestones=82,123 --seed="$SEED" --val-split=5000 --batch-size=128 --warmup-iters=0 -j 0 --save-dir="$RUN/checkpoints" --log-csv="$RUN/epoch_metrics.csv" --config-out="$RUN/config.json" --run-csv="$RUN/run_metrics.csv"
```

- Seed 0 [configuration](seed0/config.json), [epoch metrics](seed0/epoch_metrics.csv), [test metrics](seed0/run_metrics.csv), [evaluation log](seed0/final_evaluation.txt)
- Seed 1 [configuration](seed1/config.json), [epoch metrics](seed1/epoch_metrics.csv), [test metrics](seed1/run_metrics.csv), [evaluation log](seed1/final_evaluation.txt)
- [He et al., Table 6](https://arxiv.org/abs/1512.03385)


