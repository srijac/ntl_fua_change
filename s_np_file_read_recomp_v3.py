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


def main():
    
    # SETUP
    #tile_file="/Users/estokes/Desktop/RAICS/NautilusProjects/zarr_manipulate/Docker/fua_tiles_0.txt"
    # Start the clock
    stime = time()     
    print(f"Start time:{stime}.")
    # List for tiles to be downloaded
    tile_list = []
    
    # Import the list of tiles
    '''with open(Path(f"/app/examplevol-sc-covid/{tile_file}.txt"), 'r') as f:
        for line in f:
            tile_list.append(line.strip('\n'))'''
    
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
        
    w_dir_zarr=f"ceph:zarrs"
    w_dir_zarr_e=f"ceph:zarrs-east"
    w_dir_zarr_sup=f"ceph:zarrs-support"
    w_dir_zarr_sup_e=f"ceph:zarrs-support-east"
    w_dir_comb=f"ceph:vnp46a1"
    w_dir_comb_e=f"ceph:vnp46a1-east"
    w_dir_fua=f"ceph:fua-numpy"
    w_dir_fua_e=f"ceph:fua-numpy-east"
    w_dir_anom=f"ceph:anomaly-write-global"
    w_dir_anom_e=f"ceph:anomaly-write-global-east"
    w_dir_a2=f"ceph:vnp46a2"
    w_dir_a2_e=f"ceph:vnp46a2-east"
    #w_dir_list2=f"ceph:zarrs/daily_change"
    for new_dir in [w_dir_zarr, w_dir_zarr_e,w_dir_zarr_sup, w_dir_zarr_sup_e,w_dir_comb, w_dir_comb_e, w_dir_fua, w_dir_fua_e,w_dir_anom, w_dir_anom_e,w_dir_a2, w_dir_a2_e]:
        create_s3_dir(cfg, new_dir)
    
    
    #84003427_h05v05
    #poly_id='84003427'
    #tile='h05v05'
    # Load the tile -> poly dictionary
    
    with open(str(Path("/app/temp_data",f"dummy_write_bucket.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing write in east all buckets" )
    text_file.close()
    print('uncertainty')
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_zarr}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_zarr_e}/"])
                                                            
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_zarr_sup}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_zarr_sup_e}/"])
                                                            
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_comb}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_comb_e}/"])
                                                            
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_fua}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_fua_e}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_anom}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_anom_e}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_a2}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_bucket.txt")),
                                                            f"{w_dir_a2_e}/"])
    print('written')    

    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_zarr}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading zarr', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_zarr_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading zarr e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_zarr_sup}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading zarr sup', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_zarr_sup_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading zarr supb e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_comb}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading comb', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_comb_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading comb e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_fua}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading fua', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_fua_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading fua_e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_anom}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading anom', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_anom_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading anom e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_a2}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading a2', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_a2_e}/",f"dummy_write_bucket.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write_bucket.txt"),'r') as t:
        print('reading a2e', t.read())
        t.close()
        
    os.remove(str(Path(f"/app/temp_data", f"dummy_write_bucket.txt")))
        
    
        
    print('-----------')
    
    print(f"done")


if __name__ == "__main__":
    
    # Get the system argument for the tile list
    #tile_file = argv[1:][0]   
    
    #print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    #main(tile_file)
    main()
    
     