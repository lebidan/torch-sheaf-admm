# Benchmark Démineur 9x9 : Sheaf-ADMM et variantes MPNN de l'article, 3 seeds.
#
# Chaque run va dans outputs/benchmark_9x9/<variante>/seed<seed>/ ; un run dont
# checkpoint.pt existe déjà est sauté (le script peut être interrompu puis relancé).
#
# Usage (depuis la racine du dépôt) :
#   powershell -ExecutionPolicy Bypass -File scripts\run_minesweeper_benchmark.ps1 -PmStateDim 120
#   ... -Only mpnn_cm_max,mpnn_cm_max_rand -Seeds 42      # sous-ensemble
#
# -PmStateDim : taille d'état du MPNN à paramètres égaux, donnée par
#   uv run python -m scripts.match_params minesweeper_mpnn --match minesweeper_sheaf

param(
    [int]$PmStateDim = 0,
    [int[]]$Seeds = @(42, 123, 456),
    [string[]]$Only = @(),
    [string]$OutRoot = "outputs/benchmark_9x9"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path $PSScriptRoot -Parent)

# nom -> arguments Hydra (paper: CM = même d_v que le sheaf, PM = même nombre de paramètres,
# chacun en agrégation max et mean ; + MPNN à nombre de tours aléatoire, hors article)
$runs = [ordered]@{
    "sheaf"            = @("+experiment=minesweeper_sheaf")
    "mpnn_cm_max"      = @("+experiment=minesweeper_mpnn")
    "mpnn_cm_mean"     = @("+experiment=minesweeper_mpnn", "model.mpnn_aggregation=mean")
    "mpnn_pm_max"      = @("+experiment=minesweeper_mpnn", "model.agent_state_dim=$PmStateDim")
    "mpnn_pm_mean"     = @("+experiment=minesweeper_mpnn", "model.agent_state_dim=$PmStateDim", "model.mpnn_aggregation=mean")
    "mpnn_cm_max_rand" = @("+experiment=minesweeper_mpnn", "training.mpnn_train_rounds_dist=uniform", "training.mpnn_train_rounds_min=5")
}

$selected = if ($Only.Count) { $Only } else { @($runs.Keys) }
foreach ($name in $selected) {
    if (-not $runs.Contains($name)) { throw "variante inconnue : $name (connues : $($runs.Keys -join ', '))" }
    if ($name -like "mpnn_pm_*" -and $PmStateDim -le 0) {
        Write-Warning "$name sauté : donne -PmStateDim (voir scripts.match_params)."
        continue
    }
    foreach ($seed in $Seeds) {
        $dir = "$OutRoot/$name/seed$seed"
        if (Test-Path "$dir/checkpoint.pt") {
            Write-Host "déjà fait : $dir" -ForegroundColor DarkGray
            continue
        }
        Write-Host "=== $name, seed $seed -> $dir ===" -ForegroundColor Green
        $start = Get-Date
        & uv run python scripts/train.py @($runs[$name]) "training.seed=$seed" "hydra.run.dir=$dir"
        if ($LASTEXITCODE -ne 0) { throw "échec : $name seed $seed (code $LASTEXITCODE)" }
        Write-Host ("    durée : {0:N0} min" -f ((Get-Date) - $start).TotalMinutes)
    }
}

Write-Host ""
Write-Host "Comparaisons (une par variante MPNN) :" -ForegroundColor Green
$sheaf = ($Seeds | ForEach-Object { "$OutRoot/sheaf/seed$_/checkpoint_best.pt" }) -join " "
foreach ($name in $selected | Where-Object { $_ -like "mpnn_*" }) {
    $mpnn = ($Seeds | ForEach-Object { "$OutRoot/$name/seed$_/checkpoint_best.pt" }) -join " "
    Write-Host "uv run python -m scripts.compare_minesweeper --sheaf $sheaf --mpnn $mpnn --mpnn-label `"$name`" --sizes 9 11 13 15 17 19 --kmax 50 --out-dir comparisons/benchmark_9x9_$name"
}
