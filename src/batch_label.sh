#!/bin/bash

# Load variables from .env file
source ../.env

# Check if BONEPATH is set and directory exists
if [ -z "$BONEPATH" ] || [ ! -d "$BONEPATH" ]; then
    echo "BONEPATH is not set correctly or the directory does not exist."
    exit 1
fi

# Loop through all .oib files in the BONEPATH
for file in "$BONEPATH"/*.oib; do
    # Check if there are any .oib files
    if [ -e "$file" ]; then
        echo "Processing file: $file"

        # Your processing commands here
        # For example: echo "Found file $file"
    else
        echo "No .oib files found in $BONEPATH"
        break
    fi
done