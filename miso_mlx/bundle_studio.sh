#!/bin/bash
set -e

# ==============================================================================
# MisoTTS Studio macOS App Bundling Script
# ==============================================================================
# This script compiles the SPM Swift/SwiftUI application, generates a premium
# macOS-compliant App Icon (squircle with soundwave), structures a native
# .app bundle, and writes a standard Info.plist.
# ==============================================================================

PROJECT_ROOT="$(pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
APP_BUNDLE="${OUTPUT_DIR}/MisoTTS Studio.app"
CONTENTS="${APP_BUNDLE}/Contents"
MACOS_DIR="${CONTENTS}/MacOS"
RESOURCES_DIR="${CONTENTS}/Resources"

echo "=============================================================================="
echo "📦 Starting MisoTTS Studio Bundler Pipeline"
echo "=============================================================================="

# 1. Compile Swift application in Release mode
echo "🛠️  1. Compiling MisoTTS Studio SPM package (Release)..."
cd "${PROJECT_ROOT}/miso_studio"
swift build -c release
cd "${PROJECT_ROOT}"

# Locate the compiled binary
RELEASE_BIN="${PROJECT_ROOT}/miso_studio/.build/arm64-apple-macosx/release/MisoTTSStudio"
if [ ! -f "${RELEASE_BIN}" ]; then
    # Fallback in case of non-arm64 architecture pathing
    RELEASE_BIN=$(find miso_studio/.build -name MisoTTSStudio -type f | grep release | head -n 1)
fi

if [ -z "${RELEASE_BIN}" ] || [ ! -f "${RELEASE_BIN}" ]; then
    echo "❌ ERROR: Failed to locate compiled Release binary!"
    exit 1
fi
echo "✅ Compiled binary found at: ${RELEASE_BIN}"

# 2. Create native App Bundle folder hierarchy
echo "📂 2. Initializing macOS App Bundle directory structure..."
rm -rf "${APP_BUNDLE}"
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# 3. Copy the compiled executable binary
echo "🚀 3. Copying executable..."
cp "${RELEASE_BIN}" "${MACOS_DIR}/MisoTTSStudio"
chmod +x "${MACOS_DIR}/MisoTTSStudio"

# 4. Generate/Copy AppIcon.icns
echo "🎨 4. Packaging premium App Icon..."
if [ ! -f "${OUTPUT_DIR}/AppIcon.icns" ]; then
    echo "   * Generating master PNG..."
    uv run python miso_mlx/generate_app_icon.py "${OUTPUT_DIR}/master_icon.png"
    
    echo "   * Creating native .iconset..."
    mkdir -p "${OUTPUT_DIR}/AppIcon.iconset"
    sips -z 16 16     "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_16x16.png" > /dev/null 2>&1
    sips -z 32 32     "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_16x16@2x.png" > /dev/null 2>&1
    sips -z 32 32     "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_32x32.png" > /dev/null 2>&1
    sips -z 64 64     "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_32x32@2x.png" > /dev/null 2>&1
    sips -z 128 128   "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_128x128.png" > /dev/null 2>&1
    sips -z 256 256   "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_128x128@2x.png" > /dev/null 2>&1
    sips -z 256 256   "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_256x256.png" > /dev/null 2>&1
    sips -z 512 512   "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_256x256@2x.png" > /dev/null 2>&1
    sips -z 512 512   "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_512x512.png" > /dev/null 2>&1
    sips -z 1024 1024 "${OUTPUT_DIR}/master_icon.png" --out "${OUTPUT_DIR}/AppIcon.iconset/icon_512x512@2x.png" > /dev/null 2>&1
    
    echo "   * Compiling into AppIcon.icns..."
    iconutil -c icns "${OUTPUT_DIR}/AppIcon.iconset" -o "${OUTPUT_DIR}/AppIcon.icns"
    rm -rf "${OUTPUT_DIR}/AppIcon.iconset"
fi

cp "${OUTPUT_DIR}/AppIcon.icns" "${RESOURCES_DIR}/AppIcon.icns"
echo "✅ App Icon integrated successfully."

# 5. Write standard macOS CFBundle Info.plist
echo "📝 5. Writing application Info.plist configuration..."
cat <<EOF > "${CONTENTS}/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>MisoTTSStudio</string>
    <key>CFBundleIdentifier</key>
    <string>com.misotts.studio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>MisoTTS Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
</dict>
</plist>
EOF

echo "=============================================================================="
echo "🎉 SUCCESS: MisoTTS Studio has been successfully bundled!"
echo "📍 Location: ${APP_BUNDLE}"
echo "=============================================================================="

