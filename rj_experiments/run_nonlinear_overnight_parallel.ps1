$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

$envs = [System.Environment]::GetEnvironmentVariables()
$path_value = if ($envs.Contains("PATH")) { $envs["PATH"] } else { $envs["Path"] }
[System.Environment]::SetEnvironmentVariable("Path", $null, "Process")
[System.Environment]::SetEnvironmentVariable("PATH", $path_value, "Process")

$rscript = (Get-Command Rscript).Source
$out_dir = "rj_experiments/optimizer_sweep"
$log = Join-Path $repo "rj_experiments\optimizer_sweep\nonlinear_overnight_parallel.log"
$throttle = 4

function Write-RunLog {
  param([string] $Message)
  $stamp = Get-Date -Format o
  "$stamp $Message" | Tee-Object -FilePath $log -Append
}

function Add-SicnnJob {
  param(
    [System.Collections.Generic.List[object]] $Jobs,
    [string] $Block,
    [string] $Rho,
    [string] $Seed,
    [string] $Epochs,
    [string] $Lr,
    [string] $Scheduler,
    [string] $Penalty,
    [string] $Epsilon1,
    [string] $EpsilonT,
    [string] $StepsT,
    [string] $InitMode,
    [string] $HiddenScale,
    [string] $CovariateScale,
    [string] $DirectScale
  )

  $rho_tag = $Rho.Replace(".", "p")
  $pen_tag = $Penalty.Replace(".", "p")
  $lr_tag = $Lr.Replace(".", "p")
  $e1_tag = $Epsilon1.Replace(".", "p")
  $et_tag = $EpsilonT.Replace(".", "p")
  $h_tag = $HiddenScale.Replace(".", "p")
  $cov_tag = $CovariateScale.Replace(".", "p")
  $dir_tag = $DirectScale.Replace(".", "p")
  $stem = "lbbnn_nonlinear_overnight_${Block}_n1000_rho${rho_tag}_seed${Seed}_e${Epochs}_lr${lr_tag}_${Scheduler}_pen${pen_tag}_e1${e1_tag}_et${et_tag}_${InitMode}_h${h_tag}_cov${cov_tag}_dir${dir_tag}"
  $out = "$out_dir/$stem.rds"
  $stdout = "$out_dir/$stem.stdout.log"
  $stderr = "$out_dir/$stem.stderr.log"
  $args = @(
    "rj_experiments/lbbnn_nonlinear_sicnn_opt_grid.R",
    "--n-train=1000",
    "--n-test=1000",
    "--rho=$Rho",
    "--seed=$Seed",
    "--epochs=$Epochs",
    "--iter-per-epoch=10",
    "--lrs=$Lr",
    "--schedulers=$Scheduler",
    "--init-modes=$InitMode",
    "--penalty-mults=$Penalty",
    "--hidden-init-scales=$HiddenScale",
    "--covariate-init-scales=$CovariateScale",
    "--direct-init-scales=$DirectScale",
    "--epsilon-1=$Epsilon1",
    "--epsilon-T=$EpsilonT",
    "--steps-T=$StepsT",
    "--max-used-weights=200",
    "--torch-threads=1",
    "--out=$out"
  )
  $Jobs.Add([pscustomobject]@{
    Block = $Block
    Rho = $Rho
    Seed = $Seed
    Epochs = $Epochs
    Lr = $Lr
    Scheduler = $Scheduler
    Penalty = $Penalty
    Epsilon1 = $Epsilon1
    EpsilonT = $EpsilonT
    InitMode = $InitMode
    HiddenScale = $HiddenScale
    CovariateScale = $CovariateScale
    DirectScale = $DirectScale
    Out = $out
    Stdout = $stdout
    Stderr = $stderr
    Args = $args
  })
}

$jobs = New-Object System.Collections.Generic.List[object]

# Broad discovery block at rho=0. This is where exact support should be easiest.
$penalties = @("0.025", "0.05", "0.075", "0.10", "0.125", "0.15", "0.175", "0.20", "0.30")
$epsilons = @(
  @{ E1 = "0.02"; ET = "0.001"; Steps = "100" },
  @{ E1 = "0.05"; ET = "0.002"; Steps = "100" },
  @{ E1 = "0.05"; ET = "0.005"; Steps = "100" },
  @{ E1 = "0.10"; ET = "0.005"; Steps = "100" }
)
$epoch_grid = @("400", "750")
$optimizers = @(
  @{ Lr = "0.005"; Scheduler = "late" },
  @{ Lr = "0.002"; Scheduler = "mid" }
)
$init_profiles = @(
  @{ Init = "lbbnn_like"; H = "0.5"; Cov = "1"; Direct = "1" },
  @{ Init = "lbbnn_like"; H = "0.5"; Cov = "0.5"; Direct = "0.05" }
)

foreach ($epochs in $epoch_grid) {
  foreach ($opt in $optimizers) {
    foreach ($penalty in $penalties) {
      foreach ($eps in $epsilons) {
        foreach ($init in $init_profiles) {
          Add-SicnnJob `
            -Jobs $jobs `
            -Block "discover" `
            -Rho "0" `
            -Seed "20260618" `
            -Epochs $epochs `
            -Lr $opt.Lr `
            -Scheduler $opt.Scheduler `
            -Penalty $penalty `
            -Epsilon1 $eps.E1 `
            -EpsilonT $eps.ET `
            -StepsT $eps.Steps `
            -InitMode $init.Init `
            -HiddenScale $init.H `
            -CovariateScale $init.Cov `
            -DirectScale $init.Direct
        }
      }
    }
  }
}

# Validation block for the current best frontier across rho and seeds.
foreach ($penalty in @("0.125", "0.14", "0.15", "0.175", "0.20")) {
  foreach ($rho in @("0", "0.1", "0.5", "0.9")) {
    foreach ($seed in @("20260618", "20260619", "20260620")) {
      Add-SicnnJob `
        -Jobs $jobs `
        -Block "validate" `
        -Rho $rho `
        -Seed $seed `
        -Epochs "750" `
        -Lr "0.005" `
        -Scheduler "late" `
        -Penalty $penalty `
        -Epsilon1 "0.05" `
        -EpsilonT "0.005" `
        -StepsT "100" `
        -InitMode "lbbnn_like" `
        -HiddenScale "0.5" `
        -CovariateScale "1" `
        -DirectScale "1"
    }
  }
}

"START $(Get-Date -Format o) total=$($jobs.Count) throttle=$throttle" |
  Out-File -FilePath $log -Encoding utf8

$pending = New-Object System.Collections.Queue
foreach ($job in $jobs) {
  if (Test-Path -LiteralPath $job.Out) {
    Write-RunLog "SKIP block=$($job.Block) rho=$($job.Rho) seed=$($job.Seed) epochs=$($job.Epochs) lr=$($job.Lr) scheduler=$($job.Scheduler) penalty=$($job.Penalty) e1=$($job.Epsilon1) et=$($job.EpsilonT)"
  } else {
    $pending.Enqueue($job)
  }
}

$active = New-Object System.Collections.Generic.List[object]

while ($pending.Count -gt 0 -or $active.Count -gt 0) {
  while ($pending.Count -gt 0 -and $active.Count -lt $throttle) {
    $job = $pending.Dequeue()
    Write-RunLog "RUN block=$($job.Block) rho=$($job.Rho) seed=$($job.Seed) epochs=$($job.Epochs) lr=$($job.Lr) scheduler=$($job.Scheduler) penalty=$($job.Penalty) e1=$($job.Epsilon1) et=$($job.EpsilonT)"
    $proc = Start-Process `
      -FilePath $rscript `
      -ArgumentList $job.Args `
      -WorkingDirectory $repo `
      -WindowStyle Hidden `
      -RedirectStandardOutput $job.Stdout `
      -RedirectStandardError $job.Stderr `
      -PassThru
    $active.Add([pscustomobject]@{ Job = $job; Process = $proc })
  }

  Start-Sleep -Seconds 10

  for ($i = $active.Count - 1; $i -ge 0; $i--) {
    $item = $active[$i]
    $item.Process.Refresh()
    if ($item.Process.HasExited) {
      if (Test-Path -LiteralPath $item.Job.Out) {
        Write-RunLog "DONE block=$($item.Job.Block) rho=$($item.Job.Rho) seed=$($item.Job.Seed) epochs=$($item.Job.Epochs) lr=$($item.Job.Lr) scheduler=$($item.Job.Scheduler) penalty=$($item.Job.Penalty) e1=$($item.Job.Epsilon1) et=$($item.Job.EpsilonT)"
      } else {
        Write-RunLog "FAIL block=$($item.Job.Block) rho=$($item.Job.Rho) seed=$($item.Job.Seed) epochs=$($item.Job.Epochs) lr=$($item.Job.Lr) scheduler=$($item.Job.Scheduler) penalty=$($item.Job.Penalty) e1=$($item.Job.Epsilon1) et=$($item.Job.EpsilonT)"
      }
      $active.RemoveAt($i)
    }
  }
}

Write-RunLog "END"
