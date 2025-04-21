#!/bin/bash

# Create papers directory
mkdir -p paperDataset

# Extract URLs from the JSON file using grep and cut
echo "Extracting URLs from tdm_annotations.json..."
grep -o '"PaperURL": "[^"]*"' leaderboard-generation/tdm_annotations.json | cut -d'"' -f4 | while read -r url; do
    filename=$(basename "$url")
    if [ ! -f "paperDataset/$filename" ]; then
        echo "Downloading $filename..."
        curl -L "$url" -o "paperDataset/$filename"
    else
        echo "Skipping $filename (already exists)"
    fi
done 