#!/bin/bash

# Check if a directory path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Remove trailing slash
TARGET_DIR="${1%/}"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

PARENT_DIR=$(dirname "$TARGET_DIR")
FOLDER_NAME=$(basename "$TARGET_DIR")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NEW_DIR="${PARENT_DIR}/${FOLDER_NAME}_${TIMESTAMP}"

# Move the directory
mv "$TARGET_DIR" "$NEW_DIR"

echo "Moved to: $NEW_DIR"

