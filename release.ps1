<#
.SYNOPSIS
    Build and publish a Spaghetti Lab release.

.DESCRIPTION
    Bumps the version across all config files, builds the NSIS installer,
    commits, pushes, and creates a GitHub release with binaries.

.PARAMETER Version
    The version string, e.g. "2026.3.16-preview.2".
    If omitted, auto-generates YYYY.M.DD-preview.N based on today's date.

.PARAMETER SkipBuild
    Skip the Tauri build step (use existing artifacts).

.PARAMETER SkipPush
    Skip git commit/push (just build and create release from existing commit).

.EXAMPLE
    .\release.ps1
    .\release.ps1 -Version "2026.4.1-preview.1"
    .\release.ps1 -SkipBuild
#>
param(
    [string]$Version,
    [switch]$SkipBuild,
    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path $RepoRoot "app"
$TauriDir = Join-Path $AppDir "src-tauri"

# --- Resolve gh CLI ---
$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    $gh = "C:\Program Files\GitHub CLI\gh.exe"
    if (-not (Test-Path $gh)) {
        Write-Error "GitHub CLI (gh) not found. Install it: winget install GitHub.cli"
        exit 1
    }
} else {
    $gh = $gh.Source
}

# --- Determine version ---
if (-not $Version) {
    $today = Get-Date
    $datePrefix = "{0}.{1}.{2}" -f $today.Year, $today.Month, $today.Day

    # Find the latest preview number for today's date (check local tags + GitHub releases)
    $existingTags = @()
    $existingTags += git -C $RepoRoot tag -l "v${datePrefix}-preview.*" 2>$null
    $ghReleases = & $gh release list --limit 20 2>$null | Select-String "v${datePrefix}-preview\.(\d+)" -AllMatches
    foreach ($m in $ghReleases.Matches) { $existingTags += $m.Value }
    $maxN = 0
    foreach ($tag in $existingTags) {
        if ($tag -match "preview\.(\d+)") {
            $n = [int]$Matches[1]
            if ($n -gt $maxN) { $maxN = $n }
        }
    }
    $Version = "${datePrefix}-preview.$($maxN + 1)"
}

Write-Host "=== Spaghetti Lab Release ===" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Yellow

# --- Files to update ---
$tauriConf = Join-Path $TauriDir "tauri.conf.json"
$cargoToml = Join-Path $TauriDir "Cargo.toml"
$packageJson = Join-Path $AppDir "package.json"

# --- Read current version from tauri.conf.json ---
$confContent = Get-Content $tauriConf -Raw
if ($confContent -match '"version":\s*"([^"]+)"') {
    $oldVersion = $Matches[1]
} else {
    Write-Error "Could not find version in tauri.conf.json"
    exit 1
}

Write-Host "Bumping $oldVersion -> $Version" -ForegroundColor Gray

# --- Update version in all files ---
foreach ($file in @($tauriConf, $cargoToml, $packageJson)) {
    $content = Get-Content $file -Raw
    $content = $content -replace [regex]::Escape($oldVersion), $Version
    Set-Content $file $content -NoNewline
}
Write-Host "[OK] Version updated in config files" -ForegroundColor Green

# --- Build ---
if (-not $SkipBuild) {
    Write-Host "`nBuilding NSIS installer..." -ForegroundColor Cyan

    # Kill any running dev server
    Get-Process "spaghetti-lab" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    Push-Location $AppDir
    try {
        npm run tauri build -- --bundles nsis
        if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    } finally {
        Pop-Location
    }
    Write-Host "[OK] Build complete" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Build" -ForegroundColor DarkGray
}

# --- Verify artifacts exist ---
$nsisExe = Join-Path $TauriDir "target\release\bundle\nsis\Spaghetti Lab_${Version}_x64-setup.exe"
$standaloneExe = Join-Path $TauriDir "target\release\spaghetti-lab.exe"

if (-not (Test-Path $nsisExe)) {
    Write-Error "NSIS installer not found: $nsisExe"
    exit 1
}
if (-not (Test-Path $standaloneExe)) {
    Write-Error "Standalone exe not found: $standaloneExe"
    exit 1
}

$nsisSize = [math]::Round((Get-Item $nsisExe).Length / 1MB, 1)
$exeSize = [math]::Round((Get-Item $standaloneExe).Length / 1MB, 1)
Write-Host "Installer: ${nsisSize}MB  |  Standalone: ${exeSize}MB" -ForegroundColor Gray

# --- Git commit and push ---
if (-not $SkipPush) {
    Write-Host "`nCommitting and pushing..." -ForegroundColor Cyan
    Push-Location $RepoRoot
    try {
        git add app/package.json app/src-tauri/Cargo.lock app/src-tauri/Cargo.toml app/src-tauri/tauri.conf.json
        git commit -m "v${Version}"
        if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
        git push origin master
        if ($LASTEXITCODE -ne 0) { throw "Push failed" }
    } finally {
        Pop-Location
    }
    Write-Host "[OK] Pushed to origin/master" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Git push" -ForegroundColor DarkGray
}

# --- Create GitHub release ---
Write-Host "`nCreating GitHub release..." -ForegroundColor Cyan

$notes = @"
## Download

- **Spaghetti Lab_${Version}_x64-setup.exe** -- Windows installer (recommended, registers .spag file association)
- **spaghetti-lab.exe** -- Standalone executable

> This is a preview release. Expect bugs.
"@

& $gh release create "v${Version}" `
    $nsisExe `
    $standaloneExe `
    --title "v${Version}" `
    --prerelease `
    --notes $notes

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create release"
    exit 1
}

Write-Host "`n=== Released v${Version} ===" -ForegroundColor Green
Write-Host "https://github.com/TimesUp94/spaghettilab/releases/tag/v${Version}" -ForegroundColor Cyan
