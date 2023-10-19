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
#from s_zarr_to_numpy import *
import csv
from ts_main_covid_recomp import *
import pandas as pd


import sys
print('version',sys.version)

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
    with open(Path(f"/app/examplevol-sc-covid/{tile_file}.txt"), 'r') as f:
        for line in f:
            tile_list.append(line.strip('\n'))
    
    # Update
    print(f"Starting zarr conversion of {len(tile_list)} tiles. Configuring s3.")
    
    # Get s3 secrets
    with open(Path("/app/examplevol-sc-covid/s3accesskey2east.txt"), 'r') as f:
        for line in f:
            s3_access_key = str(line.strip()[4:-1])
            break
            
    with open(Path("/app/examplevol-sc-covid/s3secretkey2east.txt"), 'r') as f:
        for line in f:
            s3_secret_key = str(line.strip()[4:-1])
            break
           
    #with open(Path("/app/Training_dates.csv"), 'r') as f:
    #training_len=pd.read_csv(Path("/app/training_dates.csv"), 'r')
    
    # Form a remote configuration for rclone
    cfg = """[ceph]
    type = s3
    provider = Ceph Object Storage
    endpoint = http://rook-ceph-rgw-easts3.rook-east
    access_key_id = {0}
    secret_access_key = {1}
    region =
    nounc = true"""
    
    # Add the s3 secrets to the configuration
    cfg = cfg.format(s3_access_key, s3_secret_key)
    
    # Make s3 "directories" for the output data
    '''for new_dir in ["ceph:fua_subset_numpy"]:#Fua_run2 is the main dir? with fua as subdir?
        create_s3_dir(cfg, new_dir)'''
        
    w_dir_list=f"ceph:zarrs"
    w_dir_list2=f"ceph:zarrs/daily_change"
    for new_dir in [w_dir_list, w_dir_list2]:
        create_s3_dir(cfg, new_dir)
    
    
    #84003427_h05v05
    #poly_id='84003427'
    #tile='h05v05'
    # Load the tile -> poly dictionary
    
    with open(str(Path("/app/temp_data",f"dummy_write_{tile_file}.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing write in east" )
    text_file.close()
    print('uncertainty')
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_{tile_file}.txt")),
                                                            f"{w_dir_list}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_{tile_file}.txt")),
                                                            f"{w_dir_list2}/"])
    print('written')    

    os.remove(str(Path(f"/app/temp_data", f"dummy_write_{tile_file}.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_list}/",f"dummy_write_{tile_file}.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_{tile_file}.txt"),'r') as t:
        print('reading', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_{tile_file}.txt")))
        
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_list2}/",f"dummy_write_{tile_file}.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_{tile_file}.txt"),'r') as t:
        print('reading2', t.read())
        t.close()
        
    print('-----------')
    
    print(f"done")
    
    w_dir_list=f"ceph:test-anthrop"
    w_dir_list2=f"ceph:test-anthrop/daily_change"
    for new_dir in [w_dir_list, w_dir_list2]:
        create_s3_dir(cfg, new_dir)
        
    with open(str(Path("/app/temp_data",f"dummy_write_v2_{tile_file}.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing write in east in new bucket" )
    text_file.close()
    print('uncertainty')
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_v2_{tile_file}.txt")),
                                                            f"{w_dir_list}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_v2_{tile_file}.txt")),
                                                            f"{w_dir_list2}/"])
    print('written')    
    
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_v2_{tile_file}.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_list}/",f"dummy_write_v2_{tile_file}.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_v2_{tile_file}.txt"),'r') as t:
        print('reading', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_v2_{tile_file}.txt")))
        
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_list2}/",f"dummy_write_v2_{tile_file}.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_v2_{tile_file}.txt"),'r') as t:
        print('reading2', t.read())
        t.close()
        
    print('-----------')
    
    print(f"done")


if __name__ == "__main__":
    
    # Get the system argument for the tile list
    tile_file = argv[1:][0]   
    
    print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    main(tile_file)
    
     