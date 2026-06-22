$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

$out_dir = "rj_experiments/confirmation_runs"
$out = "$out_dir/lbbnn_linear_confirmation_post_bic_scale_20260619_results.rds"
$stdout = Join-Path $repo "$out_dir\lbbnn_linear_confirmation_post_bic_scale_20260619.stdout.log"
$stderr = Join-Path $repo "$out_dir\lbbnn_linear_confirmation_post_bic_scale_20260619.stderr.log"

New-Item -ItemType Directory -Path (Join-Path $repo $out_dir) -Force | Out-Null

$args = @(
  "rj_experiments/lbbnn_linear_sicnn_sim.R",
  "--preset=paper",
  "--rhos=0,0.1,0.5,0.9",
  "--reps=10",
  "--epochs=2000",
  "--iter-per-epoch=5",
  "--hidden-sizes=20,20,20,20",
  "--activation=sigmoid",
  "--lr=0.002",
  "--sch-step-size=500",
  "--penalty=1770.6621379746894",
  "--epsilon-1=1",
  "--epsilon-T=1e-5",
  "--steps-T=200",
  "--sic-threshold=0.5",
  "--sic-threshold-type=phi",
  "--workers=4",
  "--torch-threads=1",
  "--out=$out"
)

& (Get-Command Rscript).Source @args 1> $stdout 2> $stderr
exit $LASTEXITCODE