# push_forecasts.ps1 — Daily inference and publish to Market Dashboard
# Scheduled via Windows Task Scheduler to run once per day.
#
# What it does:
#   1. Downloads today's ECMWF ENS and runs inference for all three markets
#   2. Copies the 3 JSON files to Market-Dashboard/data/forecasts/
#   3. Commits and pushes if anything changed (Vercel redeploys automatically)

$ErrorActionPreference = "Stop"

$ForecastDir  = "C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\Price Forecasting"
$DashboardDir = "C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\Market Dashboard"
$Python       = "C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\python-projects\standard_env\Scripts\python.exe"
$LogFile      = "$ForecastDir\logs\push_forecasts.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Ensure log directory exists
New-Item -ItemType Directory -Force -Path "$ForecastDir\logs" | Out-Null

Set-Location $ForecastDir

Log "── Starting daily forecast update ────────────────────────"

# 1. Run inference
Log "Step 1: inference.py"
& $Python src/pipeline/inference.py 2>&1 | Tee-Object -Append $LogFile
if ($LASTEXITCODE -ne 0) { Log "ERROR: inference.py failed (exit $LASTEXITCODE)"; exit 1 }

# 2. Copy JSON files to Market Dashboard
Log "Step 2: copying forecasts to Market Dashboard"
$TargetDir = "$DashboardDir\data\forecasts"
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
Copy-Item "$ForecastDir\output\forecasts\*.json" $TargetDir -Force

# 3. Commit and push if anything changed
Set-Location $DashboardDir
git add data/forecasts/
$changed = & git diff --cached --name-only data/forecasts/
if ($changed) {
    Log "Step 3: committing and pushing updated forecasts"
    & git add data/forecasts/
    & git commit -m "Update forecasts $(Get-Date -Format 'yyyy-MM-dd')"
    & git push
    Log "Done — Vercel will redeploy automatically."
} else {
    Log "Step 3: forecasts unchanged — nothing to push."
}

Log "── Finished ───────────────────────────────────────────────"
