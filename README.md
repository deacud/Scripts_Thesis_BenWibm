# Scripts_Thesis
Repository containing Python scripts of my Thesis for sharing


The `preprocess_GRIB2_xxx.py` scripts are used to read in the Data from the GRIB2 output of the AROME simulations.
They use Dask in the background to deal with the large files and to not mess up the available memory they are chunked (work is done on personal Laptop using an external hard drive where GRIB files are saved).
As example `preprocess_GRIB2_hybridPressure.py` can be called from the shell and creates netcdf files for the addressed parameters for the defined model simulation.
The netcdf file are used for faster access in the Analysis afterwards.
The usage is as following: e.g. `python preprocess_GRIB2.py -r 'OP500' -p 'u' -e '10.3, 12.6, 46.8, 48.2' -s`
````
```
Usage:
       -h                      : print the help
       -r                      : modelrun to preprocess ('OP500, OP1000, OP2500, ARP1000, IFS1000')
                                  (default: 'OP500')
       -p,                     : parameters to preprocess (e.g. 'u, v, z, pres, t, tke, q')
       -e                      : extent of lon/lat grid (e.g. '10.3, 12.6, 46.8, 48.2') (optional)
       -i                      : path to GRIB files (optional)
       -s                      : save preprocessed file as netcdf (optional)
```
````
This creates the file `ds_OP500_u_hybridPressure_[10.3, 12.6, 46.8, 48.2].nc` which contains the data of zonal wind on the available hybrid pressure levels for the extent defined.
Similar things happen in the rest of the scripts but on other type of levels.

The `preprocess_Station_hybridPressure.py` script combines the netcdf Datasets on hybrid pressure levels (e.g. `ds_OP500_xxx_hybridPressure_[10.3, 12.6, 46.8, 48.2].nc`) and interpolates it to the addressed station location.
It can be used from the shell or within the code: e.g. `python preprocess_Station_hybridPressure.py -r 'OP500' -p 'u,v,pres,z' -c '11.6222,47.3053' -s` 
````
```
 Usage:
       -h                      : print the help
       -r                      : modelrun to preprocess ('OP500, OP1000, OP2500, ARP1000, IFS1000')
                                  (default: 'OP500')
       -p                      : parameters to preprocess (e.g. 'u,v,pres,..')
       -c                      : coordinates (lon, lat) e.g. '11,47'
       -i                      : path to GRIB files (optional)
       -s                      : save interpolated file as netcdf (optional)
```
````
This creates the file `ds_OP500_interp_hybridPressure_(11.6222, 47.3053).nc` which contains the data of zonal and meridional wind, pressure and geopotential on the available hybrid pressure levels interpolated to defined coordinates.
Again this is done for faster access in the Analysis afterwards (e.g. for creating profiles)







