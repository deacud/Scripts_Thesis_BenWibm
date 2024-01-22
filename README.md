# Scripts_Thesis
Repository containing Python scripts of my Thesis for sharing

The scripts are build up for the different tasks to be addressed. There is no main entry point, each script can be run on his own or is used within other scripts.
In the following some important scripts are summarized.

----
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

----


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
Again this is done for faster access in the Analysis afterwards (e.g. for creating profiles).

----


`read_RS.py` is used to read in the Radiosonde data. It can deal with the CROSSINN Radiosonde measurements as well as data retrieved from the University of Wyoming webpage. Is used within other scripts.

`read_StationMeta.py` is used to read in the station meta data defined in the Stations.csv file. Is used within other scripts.

`TS_read_stations.py` is used to read in station observations data from stations defined in Stations.csv. Is used within other scripts.

`read_VCS_lidar.py` is used to read in WLS200s lidar data from Kolsass, Hochhaeuser and Mairbach. Is used within other scripts to create vertical cross section over innvalley.

`read_lidar_vert.py` is used to read in SL88 lidar data from Kolsass. Is used within other scripts e.g. to create Height-time plot of observed wind speed over Kolsass.

`read_iBox_Fluxes.py` is used to read in Flux data from iBox stations. Is used within other scripts to analyse surface fluxes like sensible heat flux etc.

----

`path_handling.py` handles the path to files etc. **Needs to be adapted to personal need!!**

`calculations.py` script which contains equations needed.

`skewT_WB_new.py` script which does plotting of skewT diagrams and wind profiles.








