#import zarr
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
import csv
from ts_main_v3 import *
import pandas as pd
import sys



def create_s3_dir(config, uri):

    rclone.with_config(config).run_cmd(command="mkdir", extra_args=[uri, "--quiet"])


# Function to return the time difference from a start time until now
def time_diff(start_time):

    return np.around(time() - start_time, decimals=2)


def main(tile_file):
    
    # SETUP
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
           
    
    with open(Path("/app/eval.json"),'r') as jf:
        eval_dates=json.load(jf)
        
    with open(Path("/app/training_end_dates.json"),'r') as jf:
        training_len=json.load(jf)
        
    print('----training_len----:', training_len)
    
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
    w_dir="ceph:anomaly_write"
    w_dir_wt="ceph:anomaly_write/weights"
    w_dir_fc="ceph:anomaly_write/forecast"
    w_dir_comp="ceph:anomaly_write/composite"
    for new_dir in [w_dir, w_dir_wt, w_dir_fc,w_dir_comp ]:
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
            
                                                   
            existing_ds_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_obs.npy"])
            rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_obs.npy", str(Path(f"/app/temp_data/"))]) 
            
            existing_dt_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_date.npy"])
            rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"ceph:fua_subset_numpy/fua_{poly_id}_{tile}_date.npy", str(Path(f"/app/temp_data/"))])
                                            
            
        # If the zarr file exists,print
            if len(existing_ds_files_ls['out'].decode("utf-8")) > 0:
                print(f"Completed npy data file found for tile {tile} and fua {poly_id}.")
            else:
                print('not found: numpy data')
            if len(existing_dt_files_ls['out'].decode("utf-8")) > 0:
                print(f"Completed npy date file found for tile {tile} and fua {poly_id}.")
            else:
                print('not found: numpy date')
            print('looking at polygon:', poly_id)
            #do stuff to the zarr 
                
            zarr_path_data=Path("app/temp_data",f"fua_{poly_id}_{tile}_obs.npy")
            zarr_path_date=Path("app/temp_data",f"fua_{poly_id}_{tile}_date.npy")
            
            
            print('zarr o path is:',zarr_path_data)
            print('string o path:',str(zarr_path_data))
            
            print('zarr d path is:',zarr_path_date)
            print('string d path:',str(zarr_path_date))
                
            
            zarr_path_obs=Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_obs.npy")
            zarr_path_date=Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_date.npy")
            
            print('zarr path 2', zarr_path_obs, zarr_path_date)
            print('poly id:', poly_id)
            print('poly id in training len:', training_len[poly_id])
            print('trainig len:', training_len)
            end_d=training_len[poly_id][0].split('-')[0]
            end_m=training_len[poly_id][0].split('-')[1]
            end_y=training_len[poly_id][0].split('-')[2]
            print('----CALLING WITH-----', end_d, end_m, end_y)
            
            forecast_city_list(poly_id, tile, end_d,end_m,end_y, zarr_path_obs, zarr_path_date, w_dir_wt, w_dir_fc,w_dir_comp, eval_dates)

            
            print(f"done training {poly_id} gap-filled")
                
            
            
            print('directory:', w_dir_wt, w_dir_fc)
            
            #COPY CNN FILES
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiCNN_{poly_id}_{tile}_default_lr_v2.h5")),
                                                        f"{w_dir_wt}/"])
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiCNN_pred_{poly_id}_{tile}_default_lr_v2.npy")),
                                                        f"{w_dir_fc}/"])
            
            
            #COPY ANN FILES
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiANN_{poly_id}_{tile}_default_lr_v2.h5")),
                                                        f"{w_dir_wt}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiANN_pred_{poly_id}_{tile}_default_lr_v2.npy")),
                                                        f"{w_dir_fc}/"])
                                                        
            
            
            
            #COPY LSTM FILES
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"wts_multiLSTM_{poly_id}_{tile}_default_lr_with_relu_v2.h5")),
                                                        f"{w_dir_wt}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",
                                                                 f"multiLSTM_pred_{poly_id}_{tile}_default_lr_with_relu_v2.npy")),
                                                        f"{w_dir_fc}/"])
                                                        
            #COPY FINAL PREDICTIONS
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_pred_mse.csv")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_ch_pt.csv")),
                                                        f"{w_dir_comp}/"])
                                                        
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_pred_ntl_scale.csv")),
                                                        f"{w_dir_comp}/"])
                                                        
            
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_ens_pred.png")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_ens_pred_flag.png")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_ch_direction_pred.png")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_pred_ens_err.png")),
                                                        f"{w_dir_comp}/"])
                                                        
            
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_pred_err_th_v3.png")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_eval.csv")),
                                                        f"{w_dir_comp}/"])
                                                        
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_eval_markers.csv")),
                                                        f"{w_dir_comp}/"])
            
            
            rclone.with_config(cfg).run_cmd(command="copy", 
                                            extra_args=[str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_gt_marker.png")),
                                                        f"{w_dir_comp}/"])
            
            # Remove zarr file from container
            
            
            os.remove(str(Path("/app/temp_data", f"multiLSTM_pred_{poly_id}_{tile}_default_lr_with_relu_v2.npy")))
            os.remove(str(Path("/app/temp_data", f"multiANN_pred_{poly_id}_{tile}_default_lr_v2.npy")))
            os.remove(str(Path("/app/temp_data", f"multiCNN_pred_{poly_id}_{tile}_default_lr_v2.npy")))
            os.remove(str(Path("/app/temp_data", f"wts_multiLSTM_{poly_id}_{tile}_default_lr_with_relu_v2.h5")))
            os.remove(str(Path("/app/temp_data", f"wts_multiCNN_{poly_id}_{tile}_default_lr_v2.h5")))
            os.remove(str(Path("/app/temp_data", f"wts_multiANN_{poly_id}_{tile}_default_lr_v2.h5")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_pred_mse.csv")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_ch_pt.csv")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_pred_ntl_scale.csv")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_eval.csv")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_eval_markers.csv")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_ens_pred.png")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_ens_pred_flag.png")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_ch_direction_pred.png")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_pred_ens_err.png")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_pred_err_th_v3.png")))
            os.remove(str(Path("/app/temp_data", f"fua_{poly_id}_{tile}_gt_marker.png")))
                
            # Remove zarr file from container
            # Update
            print(f"Finished cleanup for {tile} {poly_id} in {np.around(time() - ptime, decimals=2)}s.") 
                
        print(f"Total time to complete: {np.around(time() - stime, decimals=2)}s.")


if __name__ == "__main__":
    
    # Get the system argument for the tile list
    tile_file = argv[1:][0]   
    
    print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    main(tile_file)
    
     