import zarr
import h5py
from pathlib import Path, PureWindowsPath
import numpy as np
from time import time, sleep
import os
import pickle
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
import rclone
import json
from shutil import rmtree
from sys import argv
from s_zarr_to_numpy import *
import csv




def chunk_files(files_list, chunk=20):
    it = iter(files_list)
    while True:
        piece = list(islice(it, chunk))
        if piece:
            yield piece
        else:
            return


def create_s3_dir(config, uri):

    rclone.with_config(config).run_cmd(command="mkdir", extra_args=[uri, "--quiet"])


# Function to return the time difference from a start time until now
def time_diff(start_time):

    return np.around(time() - start_time, decimals=2)


def main(tile_file):
    
    # SETUP
    #tile_file="/Users/estokes/Desktop/RAICS/NautilusProjects/zarr_manipulate/Docker/fua_tiles_0.txt"
    # Start the clock
    stime = time()     
    print(f"Start time:{stime}.")
    # List for tiles to be downloaded
    tile_list = []
    
    # Import the list of tiles
    with open(Path(f"/app/{tile_file}.txt"), 'r') as f:
        for line in f:
            tile_list.append(line.strip('\n'))
    
    # Update
    print(f"Starting zarr conversion of {len(tile_list)} tiles. Configuring s3.")
    
    # Get s3 secrets
    with open(Path("/app/s3/s3accesskey2.txt"), 'r') as f:
        for line in f:
            s3_access_key = str(line.strip()[4:-1])
            break
            
    with open(Path("/app/s3/s3secretkey2.txt"), 'r') as f:
        for line in f:
            s3_secret_key = str(line.strip()[4:-1])
            break
    
    # Form a remote configuration for rclone
    cfg = """[ceph]
    type = s3
    provider = Ceph Object Storage
    endpoint = http://rook-ceph-rgw-nautiluss3.rook
    access_key_id = {0}
    secret_access_key = {1}
    region =
    nounc = true"""
    
    # Add the s3 secrets to the configuration
    cfg = cfg.format(s3_access_key, s3_secret_key)
    
    # Make s3 "directories" for the output data
    for new_dir in ["ceph:fua_subset_numpy"]:#Fua_run2 is the main dir? with fua as subdir?
        create_s3_dir(cfg, new_dir)
       
    # Update
    print(f"Configured s3 in {time_diff(stime)}. Loading FUA polygon dictionaries.")
    ptime = time()   
    
    
    # Load the tile -> poly dictionary
    with open(Path("/app/tile_to_poly.json"), 'r') as f:
        tile_poly = json.load(f)
    
    # Update
    print(f"Loaded FUA polygon dictionaries {time_diff(ptime)}. Starting per-tile processing.")
    ptime = time()   
    
    #iterate through the tiles in tile list, finding the associated fuas
    for tile in tile_list:
        # Update
        print(f"Processing tile {tile}. Searching for existing files.")
        ptime = time()
        for poly_id in tile_poly[tile]:
        # Copy the zarr for each fua to the container from s3
            '''existing_files_ls = rclone.with_config(cfg).run_cmd(command="copy", 
                                                                extra_args=[f"ceph:zarrs/fua/fua_{poly_id}_{tile}.zarr/", str(Path("/app/temp_data/fua",
                                                                                                                                   f"fua_{poly_id}_{tile}.zarr"))])'''
            existing_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:zarrs/fua/fua_{poly_id}_{tile}.zarr"])
            rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"ceph:zarrs/fua/fua_{poly_id}_{tile}.zarr/", str(Path(f"/app/temp_data/"))])
                                            
            
        # If the zarr file exists,print
            if len(existing_files_ls['out'].decode("utf-8")) > 0:
                print(f"Completed zarr file found for tile {tile} and fua {poly_id}.")
            else:
                print('not copied')
            print('looking at polygon:', poly_id)
            #do stuff to the zarr 
                
            #zarr_path=Path("riacs/fua/fua_63000308_h11v07.zarr")
            #zarr_path=Path("app/temp_data/fua",f"fua_{poly_id}_{tile}.zarr")
            zarr_path=Path("app/temp_data",f"fua_{poly_id}_{tile}.zarr")
            
            
            print('zarr path is:',zarr_path)
            print('string path:',str(zarr_path))
                
            #test_files= (os.system("ls /app/temp_data/"))
            test_files= os.system("ls f""/app/temp_data/*")
            
            print('test files:',test_files)
            
            test_files= os.listdir(f"/app/temp_data/")
            
            print(' list test files :',test_files)
            
            zarr_path2=Path(f"/app/temp_data/",f"DNB_BRDF-Corrected_NTL.zarr")
            
            print('zarr path 2', zarr_path2)

            # Open the zarr file in read mode
            poly_zarr = zarr.open(zarr_path2, mode='r')
                
                
            ts_stack, dates=zarr_to_numpy(zarr_path,poly_zarr)# remove ra_ntls
                
            with open(str(Path("/app/temp_data/fua",f"fua_{poly_id}_{tile_name}_obs.npy")), 'wb') as f:
                np.save(f, ts_stack)
                
            with open(str(Path("/app/temp_data/fua",f"fua_{poly_id}_{tile_name}_date.npy")), 'wb') as f:
                np.save(f, np.asarray(poly_zarr["Dates"][0:]))
            
            np.savetxt(str(Path("/app/temp_data/fua",f"fua_{poly_id}_{tile_name}_obs2.csv")),ts_stack,fmt='%f', delimiter=',', newline='\n',header='ntl_avg, ntl_wt_avg, gf_avg, gf_wt_avg,gf_flag, ntl_flag')
                
            print(f"done converting {poly_id} zarr to numpy")
                
            # Creating zarr files for WSF and VNP46A2        
                
                
                
                
            # Copy the numpy observation array to the s3
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data/fua",
                                                                 f"fua_{poly_id}_{tile}_obs.npy")),
                                                        f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_obs.npy"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data/fua",
                                                                 f"fua_{poly_id}_{tile}_obs2.csv")),
                                                        f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_obs2.csv"])
                                                        
            # Copy the numpy date array to the s3
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data/fua",
                                                                 f"fua_{poly_id}_{tile}_date.npy")),
                                                        f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_date.npy"])                                            
            # Remove zarr file from container
                
            # Remove zarr file from container
            rmtree(str(Path("/app/temp_data/fua", f"fua_{poly_id}_{tile}_obs.npy")))
            rmtree(str(Path("/app/temp_data/fua", f"fua_{poly_id}_{tile}_obs2.csv")))
                
            # Remove zarr file from container
            rmtree(str(Path("/app/temp_data/fua", f"fua_{poly_id}_{tile}_date.npy")))
                
            # Update
            print(f"Finished cleanup for {tile} {poly_id} in {np.around(time() - ptime, decimals=2)}s.") 
                
        print(f"Total time to complete: {np.around(time() - stime, decimals=2)}s.")


if __name__ == "__main__":
    
    # Get the system argument for the tile list
    tile_file = argv[1:][0]   
    
    print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    main(tile_file)
    
     