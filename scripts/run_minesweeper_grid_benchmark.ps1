# Benchmark Démineur, protocole de l'article (labyrinthe) : Sheaf-ADMM et MPNN, 3 seeds.
#
# Agents 3x3 tous les 2 autour d'un anneau "hors plateau", K du sheaf tiré dans [15, 40],
# évaluation à 100, MPNN à 40 tours fixes, 50 époques (voir configs/experiment/minesweeper_grid_*.yaml).
# -Size 6 (défaut) : plateaux 6x6 / 5 mines, tests 8, 12, 24 ; -Size 9 : 9x9 / 10 mines, tests 13, 17, 33.
# Chaque run va dans outputs/benchmark_grid_<S>x<S>/<variante>/seed<seed>/ ; un run dont
# checkpoint.pt existe déjà est sauté (le script peut être interrompu puis relancé).
#
# Tout en une commande (depuis la racine du dépôt) :
#   powershell -ExecutionPolicy Bypass -File scripts\run_minesweeper_grid_benchmark.ps1 -Size 6
# Le script enchaîne, en sautant ce qui est déjà fait :
#   1. les tests (tests/test_minesweeper_grid.py ; -SkipTests pour les sauter) ;
#   2. les données si datasets/minesweeper_grid_<S>x<S> n'existe pas (build_minesweeper --ood-sizes) ;
#   3. la taille du MPNN à paramètres égaux (scripts.match_params --half-message), sauf si -PmStateDim est donné,
#      et celle du gros MPNN (4x les paramètres du sheaf, --scale 4), sauf si -BigStateDim est donné ;
#   4. les entraînements (variantes x seeds), puis affiche les commandes de comparaison.
# Sous-ensemble : ... -Only sheaf,mpnn_pm_max -Seeds 42   (virgules acceptées)

param(
    [ValidateSet(6, 9)][int]$Size = 6,
    [int]$PmStateDim = 0,
    [int]$BigStateDim = 0,
    [double]$BigScale = 4,
    [string[]]$Seeds = @("42", "123", "456"),
    [string[]]$Only = @(),
    [string]$OutRoot = "",
    [int]$Workers = 8,
    [switch]$SkipTests
)

# With "powershell -File", "a,b,c" arrives as one string: split on commas ourselves.
$Only = @($Only | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
    ForEach-Object { [int]$_ })

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path $PSScriptRoot -Parent)

$exp = if ($Size -eq 6) { "minesweeper_grid_6x6" } else { "minesweeper_grid" }
$testSizes = if ($Size -eq 6) { "6 8 10 12 16 24" } else { "9 13 17 25 33" }
if (-not $OutRoot) { $OutRoot = "outputs/benchmark_grid_${Size}x${Size}" }
$mines = if ($Size -eq 6) { 5 } else { 10 }
$data = "datasets/minesweeper_grid_${Size}x${Size}"

# 1. tests
if (-not $SkipTests) {
    Write-Host "=== tests ===" -ForegroundColor Green
    & uv run pytest tests/test_minesweeper_grid.py -q
    if ($LASTEXITCODE -ne 0) { throw "les tests échouent : rien n'est lancé (envoie la sortie)." }
}

# 2. données
if (-not (Test-Path "$data/test_ood_4x/all__inputs.npy")) {
    Write-Host "=== données $data ===" -ForegroundColor Green
    & uv run python -m sheaf_admm.data.build_minesweeper --size $Size --mines $mines --train-size 10000 `
        --test-size 1000 --ood-sizes --workers $Workers --output-dir $data
    if ($LASTEXITCODE -ne 0) { throw "échec de la génération des données (code $LASTEXITCODE)" }
} else {
    Write-Host "données déjà là : $data" -ForegroundColor DarkGray
}

# 3. tailles des MPNN : à paramètres égaux (PM) et gros (BIG = BigScale x les paramètres du sheaf)
function Get-MatchedStateDim([double]$scale) {
    $ErrorActionPreference = "Continue"   # PowerShell 5 turns redirected stderr lines into errors
    $out = & uv run python -m scripts.match_params "${exp}_mpnn" --match "${exp}_sheaf" --half-message --scale $scale 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $out | ForEach-Object { Write-Host "    $_" }
    if ($code -ne 0) { throw "échec de match_params (code $code)" }
    $m = [regex]::Match(($out -join "`n"), "model\.agent_state_dim=(\d+)")
    if (-not $m.Success) { throw "match_params n'a pas donné de taille (voir la sortie ci-dessus)" }
    return [int]$m.Groups[1].Value
}
$wanted = if ($Only.Count) { $Only } else { @("mpnn_pm_max", "mpnn_big_max") }
if (@($wanted | Where-Object { $_ -like "mpnn_pm_*" }).Count -and $PmStateDim -le 0) {
    Write-Host "=== taille du MPNN à paramètres égaux ===" -ForegroundColor Green
    $PmStateDim = Get-MatchedStateDim 1
    Write-Host "    -> PM : agent_state_dim = $PmStateDim"
}
if (@($wanted | Where-Object { $_ -like "mpnn_big_*" }).Count -and $BigStateDim -le 0) {
    Write-Host "=== taille du gros MPNN ($BigScale x les paramètres du sheaf) ===" -ForegroundColor Green
    $BigStateDim = Get-MatchedStateDim $BigScale
    Write-Host "    -> BIG : agent_state_dim = $BigStateDim"
}

$half = [Math]::Max(1, [int][Math]::Floor($PmStateDim / 2))
$pm = @("model.agent_state_dim=$PmStateDim", "model.message_dim=$half", "model.mpnn_message_dim=$half")
$cm = @("model.agent_state_dim=10", "model.message_dim=5", "model.mpnn_message_dim=5")
$bigHalf = [Math]::Max(1, [int][Math]::Floor($BigStateDim / 2))
$big = @("model.agent_state_dim=$BigStateDim", "model.message_dim=$bigHalf", "model.mpnn_message_dim=$bigHalf")

# nom -> arguments Hydra (article : PM = même nombre de paramètres que le sheaf, CM = même
# communication d_v / d_e ; agrégation max et mean ; hors article : PM max à nombre de tours
# aléatoire, et un gros MPNN (BIG, BigScale x les paramètres du sheaf) pour sa capacité haute)
$runs = [ordered]@{
    "sheaf"            = @("+experiment=${exp}_sheaf")
    "mpnn_pm_max"      = @("+experiment=${exp}_mpnn") + $pm
    "mpnn_pm_mean"     = @("+experiment=${exp}_mpnn", "model.mpnn_aggregation=mean") + $pm
    "mpnn_cm_max"      = @("+experiment=${exp}_mpnn") + $cm
    "mpnn_cm_mean"     = @("+experiment=${exp}_mpnn", "model.mpnn_aggregation=mean") + $cm
    "mpnn_pm_max_rand" = @("+experiment=${exp}_mpnn", "training.mpnn_train_rounds_dist=uniform",
                           "training.mpnn_train_rounds_min=15") + $pm
    "mpnn_big_max"     = @("+experiment=${exp}_mpnn") + $big
    "mpnn_big_mean"    = @("+experiment=${exp}_mpnn", "model.mpnn_aggregation=mean") + $big
}

$selected = if ($Only.Count) { $Only } else { @($runs.Keys) }
foreach ($name in $selected) {
    if (-not $runs.Contains($name)) { throw "variante inconnue : $name (connues : $($runs.Keys -join ', '))" }
    if ($name -like "mpnn_pm_*" -and $PmStateDim -le 0) {
        Write-Warning "$name sauté : donne -PmStateDim (voir scripts.match_params ... --half-message)."
        continue
    }
    if ($name -like "mpnn_big_*" -and $BigStateDim -le 0) {
        Write-Warning "$name sauté : donne -BigStateDim (voir scripts.match_params ... --scale 4)."
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
foreach ($name in $runs.Keys | Where-Object { $_ -like "mpnn_*" }) {
    $mpnn = ($Seeds | ForEach-Object { "$OutRoot/$name/seed$_/checkpoint_best.pt" }) -join " "
    Write-Host "uv run python -m scripts.compare_minesweeper --sheaf $sheaf --mpnn $mpnn --mpnn-label `"$name`" --sizes $testSizes --kmax 100 --out-dir comparisons/grid_${Size}x${Size}_$name"
}
