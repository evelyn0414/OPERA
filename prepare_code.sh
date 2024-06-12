#!/bin/bash

FILE_TO_COPY="src/model/mae_utils/swin_transformer.py" # The file you want to copy
# Find the path to timm installation
TIMM_PATH=$(python -c "import timm; print(timm.__path__[0])")

if [ -z "$TIMM_PATH" ]; then
    echo "timm is not installed or could not find the timm path"
    exit 1
fi

echo "timm path found at $TIMM_PATH"

# Copy the file to the timm path
if cp "$FILE_TO_COPY" "$TIMM_PATH/models"; then
    echo "File copied to $TIMM_PATH/models successfully"
else
    echo "Failed to copy the file"
    exit 1
fi
