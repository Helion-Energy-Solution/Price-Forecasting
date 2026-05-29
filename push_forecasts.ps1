# push_forecasts.ps1 — Daily inference and publish to Market Dashboard
# Scheduled via Windows Task Scheduler to run once per day.
#
# What it does:
#   1. Downloads fresh price CSVs from Swissgrid and refreshes price parquets
#   2. Downloads today's ECMWF ENS and runs inference for all three markets
#   3. Copies the 3 JSON files to Market-Dashboard/data/forecasts/
#   4. Commits and pushes if anything changed (Vercel redeploys automatically)

$ErrorActionPreference = "Continue"   # "Stop" treats native stderr as fatal in PS 5.1

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

Log "-- Starting daily forecast update ------------------------"

# 1. Refresh price parquets (downloads fresh CSVs from Swissgrid, then updates parquets)
Log "Step 1: refresh_prices.py (download + sync parquets)"
& $Python src/data/refresh_prices.py 2>&1 | Tee-Object -Append $LogFile
if ($LASTEXITCODE -ne 0) { Log "ERROR: refresh_prices.py failed (exit $LASTEXITCODE)"; exit 1 }

# 1b. Refresh ENTSO-E CH load/generation forecasts (non-fatal: inference falls back to
#     existing parquets / NaN if this fails or the token is missing)
Log "Step 1b: entsoe_download.py (CH load + generation forecasts)"
& $Python src/data/entsoe_download.py 2>&1 | Tee-Object -Append $LogFile
if ($LASTEXITCODE -ne 0) { Log "WARNING: entsoe_download.py failed (exit $LASTEXITCODE) - using existing parquets" }

# 2. Run inference
Log "Step 2: inference.py"
& $Python src/pipeline/inference.py 2>&1 | Tee-Object -Append $LogFile
if ($LASTEXITCODE -ne 0) { Log "ERROR: inference.py failed (exit $LASTEXITCODE)"; exit 1 }

# 3. Copy JSON files to Market Dashboard
Log "Step 3: copying forecasts to Market Dashboard"
$TargetDir = "$DashboardDir\data\forecasts"
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
Copy-Item "$ForecastDir\output\forecasts\*.json" $TargetDir -Force

# 4. Commit and push if anything changed
Set-Location $DashboardDir
git add data/forecasts/
$changed = & git diff --cached --name-only data/forecasts/
if ($changed) {
    Log "Step 4: committing and pushing updated forecasts"
    & git add data/forecasts/
    & git commit -m "Update forecasts $(Get-Date -Format 'yyyy-MM-dd')"
    & git pull --rebase 2>&1 | Tee-Object -Append $LogFile
    if ($LASTEXITCODE -ne 0) { Log "ERROR: git pull --rebase failed (exit $LASTEXITCODE)"; exit 1 }
    & git push 2>&1 | Tee-Object -Append $LogFile
    if ($LASTEXITCODE -ne 0) { Log "ERROR: git push failed (exit $LASTEXITCODE)"; exit 1 }
    Log "Done - Vercel will redeploy automatically."
} else {
    Log "Step 4: forecasts unchanged - nothing to push."
}

Log "-- Finished -----------------------------------------------"
