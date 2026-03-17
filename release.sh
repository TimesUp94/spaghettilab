#!/usr/bin/env bash
#
# Build and publish a Spaghetti Lab release.
#
# Usage:
#   ./release.sh                          # auto-version YYYY.M.DD-preview.N
#   ./release.sh 2026.4.1-preview.1       # explicit version
#   ./release.sh --skip-build             # use existing build artifacts
#   ./release.sh --skip-push              # don't commit/push
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$REPO_ROOT/app"
TAURI_DIR="$APP_DIR/src-tauri"

SKIP_BUILD=false
SKIP_PUSH=false
VERSION=""

# --- Parse args ---
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --skip-push)  SKIP_PUSH=true ;;
        --help|-h)
            echo "Usage: $0 [VERSION] [--skip-build] [--skip-push]"
            exit 0
            ;;
        *)
            if [[ -z "$VERSION" ]]; then
                VERSION="$arg"
            else
                echo "Unknown argument: $arg" >&2
                exit 1
            fi
            ;;
    esac
done

# --- Resolve gh CLI ---
if command -v gh &>/dev/null; then
    GH=gh
elif [[ -x "/c/Program Files/GitHub CLI/gh.exe" ]]; then
    GH="/c/Program Files/GitHub CLI/gh.exe"
elif [[ -x "C:/Program Files/GitHub CLI/gh.exe" ]]; then
    GH="C:/Program Files/GitHub CLI/gh.exe"
else
    echo "Error: GitHub CLI (gh) not found. Install it: winget install GitHub.cli" >&2
    exit 1
fi

# --- Determine version ---
if [[ -z "$VERSION" ]]; then
    DATE_PREFIX="$(date +%Y.%-m.%-d)"

    MAX_N=0
    # Check both local tags and GitHub releases
    while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        if [[ "$tag" =~ preview\.([0-9]+) ]]; then
            n="${BASH_REMATCH[1]}"
            (( n > MAX_N )) && MAX_N=$n
        fi
    done < <(
        git -C "$REPO_ROOT" tag -l "v${DATE_PREFIX}-preview.*" 2>/dev/null
        "$GH" release list --limit 20 2>/dev/null | grep -o "v${DATE_PREFIX}-preview\.[0-9]*" || true
    )

    VERSION="${DATE_PREFIX}-preview.$((MAX_N + 1))"
fi

echo "=== Spaghetti Lab Release ==="
echo "Version: $VERSION"

# --- Read current version ---
OLD_VERSION=$(sed -n 's/.*"version":\s*"\([^"]*\)".*/\1/p' "$TAURI_DIR/tauri.conf.json" | head -1)
if [[ -z "$OLD_VERSION" ]]; then
    echo "Error: Could not read current version from tauri.conf.json" >&2
    exit 1
fi

echo "Bumping $OLD_VERSION -> $VERSION"

# --- Update version in all files ---
for f in "$TAURI_DIR/tauri.conf.json" "$TAURI_DIR/Cargo.toml" "$APP_DIR/package.json"; do
    sed -i "s/$OLD_VERSION/$VERSION/g" "$f"
done
echo "[OK] Version updated"

# --- Signing key ---
SIGNING_KEY_PATH="$HOME/.tauri/spaghettilab.key"
if [[ -f "$SIGNING_KEY_PATH" ]]; then
    export TAURI_SIGNING_PRIVATE_KEY="$(cat "$SIGNING_KEY_PATH")"
    export TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""
    echo "[OK] Signing key loaded"
else
    echo "Warning: No signing key at $SIGNING_KEY_PATH — updater signatures will NOT be generated" >&2
fi

# --- Build ---
if [[ "$SKIP_BUILD" == false ]]; then
    echo ""
    echo "Building NSIS installer..."

    # Kill any running dev server
    taskkill //F //IM "spaghetti-lab.exe" 2>/dev/null || true

    pushd "$APP_DIR" >/dev/null
    npm run tauri build -- --bundles nsis
    popd >/dev/null

    echo "[OK] Build complete"
else
    echo "[SKIP] Build"
fi

# --- Verify artifacts ---
NSIS_EXE="$TAURI_DIR/target/release/bundle/nsis/Spaghetti Lab_${VERSION}_x64-setup.exe"
NSIS_ZIP="${NSIS_EXE}.zip"
NSIS_SIG="${NSIS_ZIP}.sig"
STANDALONE_EXE="$TAURI_DIR/target/release/spaghetti-lab.exe"

if [[ ! -f "$NSIS_EXE" ]]; then
    echo "Error: NSIS installer not found: $NSIS_EXE" >&2
    exit 1
fi
if [[ ! -f "$STANDALONE_EXE" ]]; then
    echo "Error: Standalone exe not found: $STANDALONE_EXE" >&2
    exit 1
fi

NSIS_SIZE=$(du -h "$NSIS_EXE" | cut -f1)
EXE_SIZE=$(du -h "$STANDALONE_EXE" | cut -f1)
echo "Installer: $NSIS_SIZE  |  Standalone: $EXE_SIZE"

# --- Generate latest.json for auto-updater ---
NSIS_ZIP_NAME="$(basename "$NSIS_ZIP")"
DOWNLOAD_URL="https://github.com/TimesUp94/spaghettilab/releases/download/v${VERSION}/${NSIS_ZIP_NAME}"
LATEST_JSON="$TAURI_DIR/target/release/bundle/nsis/latest.json"

if [[ -f "$NSIS_SIG" ]]; then
    SIGNATURE="$(cat "$NSIS_SIG")"
    cat > "$LATEST_JSON" <<EJSON
{
  "version": "v${VERSION}",
  "notes": "Update to v${VERSION}",
  "pub_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platforms": {
    "windows-x86_64": {
      "signature": "${SIGNATURE}",
      "url": "${DOWNLOAD_URL}"
    }
  }
}
EJSON
    echo "[OK] latest.json generated"
else
    echo "Warning: No .sig file found — latest.json NOT generated (updater won't work)" >&2
fi

# --- Git commit and push ---
if [[ "$SKIP_PUSH" == false ]]; then
    echo ""
    echo "Committing and pushing..."

    cd "$REPO_ROOT"
    git add app/package.json app/src-tauri/Cargo.lock app/src-tauri/Cargo.toml app/src-tauri/tauri.conf.json
    git commit -m "v${VERSION}"
    git push origin master

    echo "[OK] Pushed to origin/master"
else
    echo "[SKIP] Git push"
fi

# --- Create GitHub release ---
echo ""
echo "Creating GitHub release..."

NOTES="$(cat <<EOF
## Download

- **Spaghetti Lab_${VERSION}_x64-setup.exe** -- Windows installer (recommended, registers .spag file association)
- **spaghetti-lab.exe** -- Standalone executable

> This is a preview release. Expect bugs. The app will check for updates automatically.
EOF
)"

# Collect upload files
UPLOAD_FILES=("$NSIS_EXE" "$STANDALONE_EXE")
[[ -f "$NSIS_ZIP" ]] && UPLOAD_FILES+=("$NSIS_ZIP")
[[ -f "$LATEST_JSON" ]] && UPLOAD_FILES+=("$LATEST_JSON")

"$GH" release create "v${VERSION}" \
    "${UPLOAD_FILES[@]}" \
    --title "v${VERSION}" \
    --prerelease \
    --notes "$NOTES"

echo ""
echo "=== Released v${VERSION} ==="
echo "https://github.com/TimesUp94/spaghettilab/releases/tag/v${VERSION}"
