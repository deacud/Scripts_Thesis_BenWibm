#!/bin/bash

# Script that extracts the unknown pressure level variables and assigns needed information and saves it
# Author: Benedikt Wibmer, 2023-09-18

files="GRIBPFAROMAROM*.grib2"  # grib file to extract information from

rules_filter="get_uvq_pressure_rules.info"  # filter file with conditions

# loop over all files
for file in $files; do
	echo "Processing file: $file"
	hhhh_mm=${file:15:7}
	out_name="GRIBPFAROMAROM_isobaricInhPa_uvq+${hhhh_mm}.grib2"  # output name of file
	echo $rules_filter
	grib_filter -o $out_name $rules_filter $file
	
done	

