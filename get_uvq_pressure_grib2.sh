#!/bin/bash

# Script that extracts the unknown pressure level variables and assigns needed information and saves it
# Author: Benedikt Wibmer, 2023-09-18

iodir="/perm/aut0883/claef1k/DATA/20190912/12/MEM_00/GRIB2"
files="${iodir}/GRIBPFAROMAROM1k*.grib2"  # grib file to extract information from

rules_filter="get_uvq_pressure_rules.info"  # filter file with conditions

# loop over all files
for file in $files; do
    echo "Processing file: $file"
    f=$(basename "$file")
    hhhh_mm=${f:15:7}
    out_name="${iodir}/GRIBPFAROMAROM1k_isobaricInhPa_uvq+${hhhh_mm}.grib2"  # output name of file
    echo $rules_filter
    grib_filter -o $out_name $rules_filter $file	
done	

