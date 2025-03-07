#!/bin/bash

# Define the directory containing the old filenames
DIR="/data1/crops/dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA/1"

# Loop through each file in the directory
for file in "$DIR"/*.tif; do
    # Extract components from the old filename using regex
    filename=$(basename -- "$file")
    
    if [[ $filename =~ ([0-9]{8})_([0-9]{2}):([0-9]{2})_EXPATS_([0-9]+)_([0-9]+K-[0-9]+K)_([a-zA-Z]+)_CMA\.(tif) ]]; then
        DATE=${BASH_REMATCH[1]}       # Extract YYYYMMDD
        TIME="${BASH_REMATCH[2]}:${BASH_REMATCH[3]}"       # Extract HH:MM (time)
        UL_LAT="50.0"              # Set the upper left lat, this is assumed and should be adjusted
        UL_LON="6.5"              # Set the upper left lon, this is assumed and should be adjusted
        PIXEL_SIZE="0.04"             # Set pixel size
        X_PIXELS="200"                # Set X pixels
        Y_PIXELS="200"                # Set Y pixels
        BT_SCALE=${BASH_REMATCH[5]}   # Extract BT SCALE (200K-300K)
        COLOR_SCALE=${BASH_REMATCH[6]} # Extract COLOR SCALE (greyscale)
        CMA_INFO="closed-CMA"                # Fixed value for CMA info
        EXT=${BASH_REMATCH[7]}        # Extract file extension (tif)

        # Construct the new filename
        NEW_FILENAME="${DATE}-${TIME}_${UL_LAT}_${UL_LON}_${PIXEL_SIZE}_${X_PIXELS}x${Y_PIXELS}_${BT_SCALE}_${COLOR_SCALE}_${CMA_INFO}.${EXT}"
        #echo "Renamed: $filename -> $NEW_FILENAME"
         
        # # Rename the file
        mv "$file" "$DIR/$NEW_FILENAME"
        echo "Renamed: $filename -> $NEW_FILENAME"
    else
        echo "Skipping: $filename (format not recognized)"
    fi
done

echo "Filename conversion completed!"

