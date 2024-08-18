#!/bin/bash

# Script to remove .DS_Store files and __pycache__ directories

# Check if a directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Navigate to the specified directory
cd "$1" || exit

# Find and remove all .DS_Store files recursively
find . -name '.DS_Store' -type f -print -delete

# Find and remove all __pycache__ directories recursively
find . -type d -name '__pycache__' -print -exec rm -rf {} +

echo "Cleanup completed in directory: $1"
