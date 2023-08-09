#!/bin/bash

# Directory containing SVG files
dir="./transform/decks"

# Find all SVG files in the directory and its subdirectories
find "$dir" -name "*.svg" -type f | while read -r svg_file; do
  # Use qlmanage to render the SVG to PNG
  qlmanage -t -s 80 -o "$(dirname "$svg_file")" "$svg_file"
  # Rename the output file to .png
  mv "${svg_file}.png" "${svg_file%.svg}.png"
done