#!/bin/bash

# Load variables from .env file
source ../.env

# Check if BONEPATH is set and directory exists
if [ -z "$BONEPATH" ] || [ ! -d "$BONEPATH" ]; then
    echo "BONEPATH is not set correctly or the directory does not exist."
    exit 1
fi

# Loop through all .oib files in the BONEPATH
for file in $BONEPATH/weijie_selected/main_folder/FV10__oibs/*.oib; do
    # Check if there are any .oib files
    if [ -e "$file" ]; then
        echo "Processing file: $file"

	python $ROOTPATH/src/Ripley.py \
		--oib_path $file \
		--mask_path $BONEPATH/masks \
		--tumor_path $BONEPATH/tumor_locations_02_08_2023.csv \
		--branch_path $BONEPATH/branch_points \
		--ng2_path $BONEPATH/NG2_Centroids \
		--tvc_path $BONEPATH/tortuous_segment_centroids
    else
        echo "No .oib files found in $BONEPATH/weijie_selected/main_folder/FV10__oibs/"
        break
    fi
done
