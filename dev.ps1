# Launch Spaghetti Lab in dev mode.
# Kills any leftover processes first to avoid double windows / port conflicts.

# Kill leftover instances
Get-Process -Name "spaghetti-lab" -ErrorAction SilentlyContinue | Stop-Process -Force

# Free port 1420 if occupied
$conn = Get-NetTCPConnection -LocalPort 1420 -ErrorAction SilentlyContinue | Where-Object State -eq Listen
if ($conn) {
    Write-Host "Killing process $($conn.OwningProcess) on port 1420"
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

Set-Location "$PSScriptRoot\app"
npx tauri dev
