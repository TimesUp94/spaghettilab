# Our own build of Spaghetti Lab: the Lab tab can also run the research models on this PC, with no sign-in and
# nothing spent. Published builds are compiled without this feature. See docs/desktop-login.md, "Local build".
#   .\build-local.ps1          dev run (npx tauri dev --features local-models)
#   .\build-local.ps1 -Build   a standalone exe, src-tauri\target\release\spaghetti-lab.exe (not an installer)
# The models come from G:\Projects\replanal-data\ggst_dataset\model; GGST_DATA, GGST_MODEL_DIR, SPAGHETTI_LAB_MODELS or the
# Lab panel's settings override it. The AI summary uses ANTHROPIC_API_KEY when it's set.
param([switch]$Build)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ($Build) {
        npx tauri build --no-bundle --features local-models
    } else {
        npx tauri dev --features local-models
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
