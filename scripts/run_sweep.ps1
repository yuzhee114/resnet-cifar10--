<#
.SYNOPSIS
    Run the Table 6 sweep (5 models x 3 seeds) for the MIAE01 replication.

.DESCRIPTION
    Every run writes three artefacts under logs/ :
        <arch>_seed<S>.csv   one row per epoch
        <arch>_seed<S>.json  resolved configuration
        runs.csv             one row per finished run (aggregated by summarize.py)

    A run is skipped when its per-epoch CSV already contains --Epochs rows, so
    the script is safe to restart after an interruption.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/run_sweep.ps1
    powershell -ExecutionPolicy Bypass -File scripts/run_sweep.ps1 -Arches resnet20 -Seeds 0
#>
param(
    [string[]]$Arches = @("resnet20", "resnet32", "resnet44", "resnet56", "resnet110"),
    [int[]]$Seeds = @(0, 1, 2),
    [int]$Epochs = 164,
    [string]$Milestones = "82,123",
    [int]$WarmupIters = 0,
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path logs, save_runs | Out-Null

foreach ($arch in $Arches) {
    foreach ($seed in $Seeds) {
        $csv = "logs/${arch}_seed${seed}.csv"

        if (Test-Path $csv) {
            $done = (Get-Content $csv | Measure-Object -Line).Lines - 1
            if ($done -ge $Epochs) {
                Write-Host "[skip] $arch seed=$seed already has $done epochs"
                continue
            }
        }

        # the paper warms up ResNet-110/1202 for ~400 iterations
        $warmup = $WarmupIters
        if ($warmup -eq 0 -and $arch -in @("resnet110", "resnet1202")) {
            $warmup = 400
        }

        Write-Host "[run ] $arch seed=$seed epochs=$Epochs warmup=$warmup"
        python trainer.py `
            --arch=$arch `
            --epochs=$Epochs `
            --lr-milestones=$Milestones `
            --warmup-iters=$warmup `
            --seed=$seed `
            --val-split=5000 `
            -j $Workers `
            --save-dir="save_runs/${arch}_s${seed}" `
            --log-csv=$csv `
            --config-out="logs/${arch}_seed${seed}.json" `
            --run-csv logs/runs.csv

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[fail] $arch seed=$seed exited with $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
}

Write-Host "sweep finished, aggregating"
python scripts/summarize.py
