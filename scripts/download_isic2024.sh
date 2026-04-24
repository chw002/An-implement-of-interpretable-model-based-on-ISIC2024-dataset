#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Download the ISIC 2024 permissive-training release into `data/raw/isic2024/`.
# Prefers Kaggle; falls back to a friendly error if no credentials are set.
# ------------------------------------------------------------------------------
set -euo pipefail

TARGET="${1:-data/raw/isic2024_official}"
mkdir -p "$TARGET"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found.  Install via: pip install kaggle"
    exit 1
fi

TOKEN="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
if [[ ! -f "$TOKEN" ]]; then
    echo "Kaggle API token not found at $TOKEN." >&2
    echo "Create one at https://www.kaggle.com/settings/account and place it at that path," >&2
    echo "then rerun this script." >&2
    exit 2
fi

SLUG="${ISIC2024_SLUG:-tomooinubushi/isic-2024-challenge-permissive}"
echo "Downloading $SLUG into $TARGET …"
kaggle datasets download -d "$SLUG" -p "$TARGET" --unzip

echo "Done.  Expected layout:"
ls -la "$TARGET" | head
