[CmdletBinding()]
param(
    [string]$OnlineRoot = "D:\T3S_exp\AtomicSkill-ToolGraph\runs\all_small_post_acquire_discovery_fix",
    [string]$EvalRoot = "D:\T3S_exp\AtomicSkill-ToolGraph\runs\all_small_frozen_post_acquire_discovery_fix",
    [int]$AlfWorldLimit = 10,
    [int]$HumanEvalLimit = 10,
    [int]$Gsm8kLimit = 50,
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"
$Host.UI.RawUI.WindowTitle = "AtomicSkillGraph - all_small monitor"

$onlineConditions = @(
    "baseline_dynamic", "flowevo", "atomic_graph_only",
    "tool_repo_only", "atomic_skillgraph_full"
)
$frozenConditions = @(
    "atomic_graph_only", "tool_repo_only", "atomic_skillgraph_full"
)
$specs = @(
    [pscustomobject]@{ Benchmark = "alfworld"; Limit = $AlfWorldLimit },
    [pscustomobject]@{ Benchmark = "humaneval"; Limit = $HumanEvalLimit },
    [pscustomobject]@{ Benchmark = "gsm8k"; Limit = $Gsm8kLimit }
)

function Read-JsonSafe([string]$Path) {
    try {
        if (Test-Path -LiteralPath $Path) {
            return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        }
    } catch {}
    return $null
}

function Get-LatestDirectory([string]$Root, [string]$Pattern) {
    if (-not (Test-Path -LiteralPath $Root)) { return $null }
    return Get-ChildItem -LiteralPath $Root -Directory -Filter $Pattern |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Get-ConditionStats(
    [System.IO.DirectoryInfo]$RunDir,
    [string]$Condition,
    [int]$Limit,
    [bool]$Frozen
) {
    $episodes = @()
    $updated = $null

    if ($null -ne $RunDir) {
        $metricDir = Join-Path $RunDir.FullName "$Condition\data\metrics"
        if (Test-Path -LiteralPath $metricDir) {
            $metricFiles = @(Get-ChildItem -LiteralPath $metricDir -File -Filter "episode_*.json")
            foreach ($file in $metricFiles) {
                $item = Read-JsonSafe $file.FullName
                if ($null -ne $item) { $episodes += $item }
            }
            if ($metricFiles.Count -gt 0) {
                $updated = ($metricFiles | Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1).LastWriteTime
            }
        } elseif (-not $Frozen) {
            $baselineDirName = if ($Condition -eq "baseline_dynamic") {
                "baseline_dynamic_flowevo"
            } else { "flowevo_flowevo" }
            $baselineDir = Join-Path $RunDir.FullName $baselineDirName
            $checkpoint = Get-ChildItem -LiteralPath $baselineDir -File `
                -Filter "_checkpoint_*.json" |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($null -ne $checkpoint) {
                $data = Read-JsonSafe $checkpoint.FullName
                if ($null -ne $data) { $episodes = @($data.episodes) }
                $updated = $checkpoint.LastWriteTime
            }
        }
    }

    $count = [math]::Min($Limit, @($episodes).Count)
    $success = @($episodes | Where-Object {
        $_.success -eq $true -or $_.passed -eq $true
    }).Count
    $infra = @($episodes | Where-Object {
        $_.infrastructure_failure -eq $true -or $_.failure_type -eq "llm_error"
    }).Count
    $taskFailure = @($episodes | Where-Object {
        ($_.task_failure -eq $true) -or
        (($_.success -ne $true) -and ($_.passed -ne $true) -and
         ($_.infrastructure_failure -ne $true) -and ($_.failure_type -ne "llm_error"))
    }).Count
    $status = if ($count -ge $Limit) { "Complete" }
              elseif ($count -gt 0) { "Running" }
              else { "Waiting" }

    return [pscustomobject]@{
        Condition = $Condition
        Done = $count
        Total = $Limit
        Success = $success
        TaskFailure = $taskFailure
        InfraError = $infra
        Status = $status
        Updated = $updated
    }
}

function New-Bar([double]$Fraction, [int]$Width = 36) {
    $ratio = [math]::Max(0.0, [math]::Min(1.0, $Fraction))
    $filled = [math]::Floor($ratio * $Width)
    return ("#" * $filled) + ("-" * ($Width - $filled))
}

function Get-StageRows {
    $rows = @()
    foreach ($spec in $specs) {
        $onlineRun = Get-LatestDirectory $OnlineRoot "$($spec.Benchmark)_*"
        $onlineStats = @()
        foreach ($condition in $onlineConditions) {
            $onlineStats += Get-ConditionStats $onlineRun $condition $spec.Limit $false
        }
        $onlineDone = ($onlineStats | Measure-Object -Property Done -Sum).Sum
        if ($null -eq $onlineDone) { $onlineDone = 0 }
        $onlineTotal = $spec.Limit * $onlineConditions.Count
        $onlineStatus = if ($onlineDone -ge $onlineTotal) { "Complete" }
                        elseif ($onlineDone -gt 0 -or $null -ne $onlineRun) { "Running" }
                        else { "Waiting" }
        $rows += [pscustomobject]@{
            Benchmark = $spec.Benchmark
            Phase = "online"
            Done = [int]$onlineDone
            Total = $onlineTotal
            Status = $onlineStatus
            Run = $onlineRun
            Details = $onlineStats
        }

        $evalRun = $null
        if ($null -ne $onlineRun) {
            $evalRun = Get-LatestDirectory $EvalRoot "$($onlineRun.Name)_train_*"
        }
        $evalStats = @()
        foreach ($condition in $frozenConditions) {
            $evalStats += Get-ConditionStats $evalRun $condition $spec.Limit $true
        }
        $evalDone = ($evalStats | Measure-Object -Property Done -Sum).Sum
        if ($null -eq $evalDone) { $evalDone = 0 }
        $evalTotal = $spec.Limit * $frozenConditions.Count
        $evalStatus = if ($evalDone -ge $evalTotal) { "Complete" }
                      elseif ($evalDone -gt 0 -or $null -ne $evalRun) { "Running" }
                      else { "Waiting" }
        $rows += [pscustomobject]@{
            Benchmark = $spec.Benchmark
            Phase = "frozen"
            Done = [int]$evalDone
            Total = $evalTotal
            Status = $evalStatus
            Run = $evalRun
            Details = $evalStats
        }
    }
    return $rows
}

while ($true) {
    $stages = @(Get-StageRows)
    $done = ($stages | Measure-Object -Property Done -Sum).Sum
    $total = ($stages | Measure-Object -Property Total -Sum).Sum
    if ($null -eq $done) { $done = 0 }
    $fraction = if ($total -gt 0) { $done / $total } else { 0.0 }
    $percent = [math]::Round(100 * $fraction, 2)
    $current = $stages | Where-Object { $_.Done -lt $_.Total } | Select-Object -First 1

    Clear-Host
    Write-Host "AtomicSkillGraph all_small experiment monitor" -ForegroundColor Cyan
    Write-Host ("=" * 86) -ForegroundColor DarkCyan
    $overallLine = "Overall [{0}] {1}/{2}  {3}%" -f `
        (New-Bar $fraction), $done, $total, $percent
    Write-Host $overallLine -ForegroundColor Green
    Write-Host "Online root: $OnlineRoot"
    Write-Host "Frozen root: $EvalRoot"
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')   Refresh: ${RefreshSeconds}s"
    Write-Host ""

    $stageView = foreach ($stage in $stages) {
        $stageFraction = if ($stage.Total -gt 0) { $stage.Done / $stage.Total } else { 0 }
        [pscustomobject]@{
            Benchmark = $stage.Benchmark
            Phase = $stage.Phase
            Progress = ("{0}/{1} ({2,6:N2}%)" -f $stage.Done, $stage.Total,
                        (100 * $stageFraction))
            Status = $stage.Status
            Run = if ($null -ne $stage.Run) { $stage.Run.Name } else { "-" }
        }
    }
    Write-Host "Six-stage progress" -ForegroundColor Yellow
    $stageView | Format-Table -AutoSize | Out-String | Write-Host

    if ($null -ne $current) {
        $currentLine = "Current stage: {0} / {1}" -f $current.Benchmark, $current.Phase
        Write-Host $currentLine -ForegroundColor Yellow
        $current.Details |
            Select-Object Condition, Done, Total, Success, TaskFailure, InfraError, Status, Updated |
            Format-Table -AutoSize | Out-String | Write-Host
    } else {
        Write-Host "All online evolution and frozen replay stages are complete." -ForegroundColor Green
        break
    }

    Write-Host "Press Ctrl+C to close this monitor; the main experiment is unaffected." -ForegroundColor DarkGray
    if ($Once) { break }
    Start-Sleep -Seconds ([math]::Max(1, $RefreshSeconds))
}

if (-not $Once) {
    Write-Host "Monitoring complete. Press Enter to close." -ForegroundColor Green
    [void](Read-Host)
}
