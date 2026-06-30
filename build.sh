#!/usr/bin/env bash
#
# Build the Metal cloth in release and wrap it into a proper .app bundle, so it
# launches as a normal foreground GUI app (Dock icon, focus, keyboard) instead
# of a bare command-line executable.
#
#   ./build.sh        build only        -> build/ClothMetal.app
#   ./build.sh run    build and launch
#   ./build.sh clean  remove build artifacts
#
set -euo pipefail

APP_NAME="ClothMetal"
CONFIG="release"
APP="build/${APP_NAME}.app"

cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf .build build
    echo "cleaned."
    exit 0
fi

if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: this is a macOS / Metal project — build on a Mac." >&2
    exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
    echo "error: swift toolchain not found. Install Xcode or the Swift command-line tools." >&2
    exit 1
fi

echo "==> swift build -c ${CONFIG}"
swift build -c "${CONFIG}"

BIN_DIR="$(swift build -c "${CONFIG}" --show-bin-path)"
BIN="${BIN_DIR}/${APP_NAME}"
if [[ ! -x "${BIN}" ]]; then
    echo "error: built executable not found at ${BIN}" >&2
    exit 1
fi

echo "==> packaging ${APP}"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS"
cp "${BIN}" "${APP}/Contents/MacOS/${APP_NAME}"

cat > "${APP}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key><string>Cloth — Metal</string>
    <key>CFBundleIdentifier</key><string>com.makarov.clothmetal</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>${APP_NAME}</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
PLIST

echo "==> built ${APP}"

if [[ "${1:-}" == "run" ]]; then
    echo "==> launching"
    open "${APP}"
fi
