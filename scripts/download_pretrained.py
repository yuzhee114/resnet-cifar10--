#!/usr/bin/env python
"""Download the pretrained CIFAR-10 ResNet checkpoints released with the paper code.

Why this script exists
----------------------
The upstream repository (akamaster/pytorch_resnet_cifar10) commits its weights,
which makes a fresh clone 86 MB and puts a 75 MB blob (`resnet1202`) into the
history - above GitHub's 50 MB warning limit and above the 50 MB commit rule of
the assignment. In this replication the `pretrained_models/` directory is
untracked and ignored, so the weights are fetched on demand by this script.

Usage
-----
    python scripts/download_pretrained.py                 # the five useful models
    python scripts/download_pretrained.py --models all    # include resnet1202
    python scripts/download_pretrained.py --models resnet20 resnet56
    python scripts/download_pretrained.py --force         # re-download

Every file is checked against the size published by GitHub; a truncated
download therefore fails loudly instead of producing a corrupt checkpoint.
"""
import argparse
import os
import shutil
import sys
import urllib.request

BASE_URL = ("https://raw.githubusercontent.com/akamaster/"
            "pytorch_resnet_cifar10/master/pretrained_models")

# model name -> (file name, expected size in bytes)
MODELS = {
    "resnet20":   ("resnet20-12fca82f.th",   1099231),
    "resnet32":   ("resnet32-d509ac18.th",   1890043),
    "resnet44":   ("resnet44-014dd654.th",   2680855),
    "resnet56":   ("resnet56-4bfd9763.th",   3471733),
    "resnet110":  ("resnet110-1d1ed7c2.th",  7031437),
    "resnet1202": ("resnet1202-f3b1deed.th", 79020193),
}
DEFAULT_MODELS = ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110"]


def human(n):
    """Format a byte count for the console."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0


def download(url, dst, expected_size, retries=3):
    """Fetch `url` into `dst`, verifying the size; retries on transient errors."""
    tmp = dst + ".part"
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mia01-replication"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(tmp, "wb") as handle:
                    shutil.copyfileobj(response, handle)

            size = os.path.getsize(tmp)
            if expected_size and size != expected_size:
                raise IOError("size mismatch: got {} bytes, expected {}".format(
                    size, expected_size))

            os.replace(tmp, dst)   # atomic on the same filesystem
            return size
        except Exception as exc:                      # noqa: BLE001 - report and retry
            print("  attempt {}/{} failed: {}".format(attempt, retries, exc))
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="pretrained_models",
                        help="destination directory (default: pretrained_models)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODELS) + ["all"],
                        help="which checkpoints to fetch (default: all but resnet1202)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a file of the right size exists")
    args = parser.parse_args()

    wanted = list(MODELS) if "all" in args.models else args.models
    os.makedirs(args.out_dir, exist_ok=True)

    print("downloading into {}".format(os.path.abspath(args.out_dir)))
    failed = []
    for name in wanted:
        filename, expected = MODELS[name]
        dst = os.path.join(args.out_dir, filename)

        if os.path.exists(dst) and not args.force:
            size = os.path.getsize(dst)
            if size == expected:
                print("  {:<10} {:<26} ok, already present ({})".format(
                    name, filename, human(size)))
                continue
            print("  {:<10} {} exists with wrong size, re-downloading".format(name, filename))

        url = "{}/{}".format(BASE_URL, filename)
        try:
            size = download(url, dst, expected)
            print("  {:<10} {:<26} ok ({})".format(name, filename, human(size)))
        except Exception as exc:                      # noqa: BLE001
            print("  {:<10} {:<26} FAILED: {}".format(name, filename, exc))
            failed.append(name)

    print("")
    if failed:
        print("failed: {}".format(", ".join(failed)))
        return 1
    print("all requested checkpoints are in {}".format(os.path.abspath(args.out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
