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
#from ts_main_covid_recomp import *
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
        
    '''w_dir_list=f"ceph:zarrs"
    w_dir_list2=f"ceph:zarrs/daily_change"
    for new_dir in [w_dir_list, w_dir_list2]:
        create_s3_dir(cfg, new_dir)
    
    
    with open(str(Path("/app/temp_data",f"dummy_write_mod.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing more writes in east" )
    text_file.close()
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_mod.txt")),
                                                            f"{w_dir_list}/"])
                                                            
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write_mod.txt")),
                                                            f"{w_dir_list2}/"])
    print('written')      
    w_dir_ds=f"ceph:anthrop-test"
    w_dir_dc_dash=f"ceph:anthrop-test/disruption/daily_change"
    
    for new_dir in [w_dir_ds,w_dir_dc_dash ]:
        create_s3_dir(cfg, new_dir)
        
    with open(str(Path("/app/temp_data",f"dummy_write2.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing write in east dash" )
    text_file.close()
    print('uncertainty')
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write2.txt")),
                                                            f"{w_dir_ds}/"])
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write2.txt")),
                                                            f"{w_dir_dc_dash}/"])
    
    os.remove(str(Path(f"/app/temp_data", f"dummy_write2.txt")))
    
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_ds}/",f"dummy_write2.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write2.txt"),'r') as t:
        print('reading', t.read())
        t.close()
        
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"{w_dir_dc_dash}/",f"dummy_write2.txt")),
                                                            f"/app/temp_data/"])
                                                            
    with open((f"/app/temp_data/dummy_write2.txt"),'r') as t:
        print('reading2', t.read())
        t.close()'''
        
    '''w_dir_covid=f"ceph:anthropause"
    w_dir_blackmarble=f"ceph:vnp46a2"
    
    for new_dir in [w_dir_covid,w_dir_blackmarble ]:
        create_s3_dir(cfg, new_dir)'''
        
    print('-----------')
    
    print('---------------')
    w_dir_us=f"ceph:anthropause/test3"
    w_dir_dis=f"ceph:anthropause/test3/disruption"
    w_dir_dc=f"ceph:anthropause/test3/disruption/daily_change"
    w_dir_cseg=f"ceph:anthropause/test3/disruption/change_segment"
    w_dir_qa=f"ceph:anthropause/test3/disruption/daily_qa_flags"
    w_dir_unc=f"ceph:anthropause/test3/disruption/city_uncertainty"
    w_dir_rec=f"ceph:anthropause/test3/recovery"
    for new_dir in [w_dir_us]:
        create_s3_dir(cfg, new_dir)
    print('made test')
        
    for new_dir in [w_dir_dis]:
        create_s3_dir(cfg, new_dir)
    print('made dis')
    for new_dir in [w_dir_dc]:
        create_s3_dir(cfg, new_dir)
    print('made dc')
    for new_dir in [w_dir_cseg]:
        create_s3_dir(cfg, new_dir)
    print('made cseg')
    for new_dir in [w_dir_qa]:
        create_s3_dir(cfg, new_dir)
    print('made qa')
    for new_dir in [w_dir_unc]:
        create_s3_dir(cfg, new_dir)
    print('made unc')
    for new_dir in [w_dir_rec]:
        create_s3_dir(cfg, new_dir)
    print('made rec')
    '''with open(str(Path("/app/temp_data",f"dummy_write3.txt")), "w") as text_file:
        #text_file.write("coeff of var, pv, cdi: %0.6f %0.6f %0.6f" % (cv, pv, cdi))
        text_file.write("testing write in east" )
    text_file.close()
    print('uncertainty')
    rclone.with_config(cfg).run_cmd(command="copy", 
                                                extra_args=[str(Path(f"/app/temp_data",f"dummy_write3.txt")),
                                                            f"{w_dir_us}/"])'''
    print('reach east end')
    
    rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"/app/examplevol-sc-covid/tile_to_poly_test.json", 
                                            f"/app/temp_data/"])
                                            
    rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"/app/examplevol-sc-covid/multitile_to_poly.json", 
                                            f"/app/temp_data/"])
    
    #with open('/app/temp_data/tile_list_gtiff_jt_reg_gen.json', 'r') as f:
    # d=json.load(f)
    with open(Path("/app/temp_data/tile_to_poly_test.json"), 'r') as f:
        tile_poly = json.load(f)
        
    with open(Path("/app/temp_data/multitile_to_poly.json"), 'r') as f:
        multitile_poly = json.load(f)
        
    
    print(tile_poly)
    
    rclone.with_config(cfg).run_cmd(command="copy",
                                            extra_args=[f"ceph:anthropause/data/composite_allmet/fua_35605728_h25v06_time_step_v2_recomp.csv", str(Path(f"/app/temp_data/"))])                            
            
    
    #fua_35605728_h25v06_time_step_v2_recomp.csv
    
    ts_test=pd.read_csv(f"/app/temp_data/fua_35605728_h25v06_time_step_v2_recomp.csv")
    
    print('test read:', ts_test)
    
    #iterate through the tiles in tile list, finding the associated fuas
    for tile in tile_list:
        # Update
        print(f"Processing tile {tile}. Searching for existing files.")
        ptime = time()
        for poly_id in tile_poly[tile]:
        # Copy the files for each fua to the container from s3
            print('tile, poly', poly_id, tile)
            
            #not in the multi tile list, so single tile, so process
            single_flag=0
            polyid_flag=0
            #not in the multi tile list, so single tile, so process
            if (tile not in multitile_poly.keys()):
                print(f'processing, {tile}, not in multi')
                single_flag=1
                print('----------------------')
            elif (((single_flag==0)&(poly_id not in multitile_poly[tile]))):#in the multi tile list, but not a multi fua, so process
                print(f'processing, {tile},{poly_id} not in multi')
                polyid_flag=1
                print('----------------------')
            elif (((single_flag==0)&(poly_id in multitile_poly[tile]))):#in the multi tile list, but not a multi fua, so process
                print(f'skipping, {tile},{poly_id} in multi')
                polyid_flag=0
                print(f'flags flags: {polyid_flag},{single_flag}')
                print('----------------------')
            if ((polyid_flag==1) | (single_flag==1)):
                print("PROCESSING, with flags", poly_id, tile, polyid_flag, single_flag)
                             
                #time-step v2
                existing_ts_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:anthropause/test3/disruption/daily_change/change_time_step_{poly_id}_{tile}.csv"])
                '''rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:zarrs/visual/composite_allmet/fua_{poly_id}_{tile}_time_step_v2_recomp.csv", str(Path(f"/app/temp_data/"))])'''
                rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:anthropause/data/composite_allmet/fua_{poly_id}_{tile}_time_step_v2_recomp.csv", str(Path(f"/app/temp_data/"))])                            
                                                    
                #change seg
                existing_seg_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:anthropause/test3/disruption/change_segment/segment_{poly_id}_{tile}.csv"])
                '''rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:zarrs/visual/composite_allmet_segment/fua_{poly_id}_{tile}_segment_v2_obs_recomp.csv", str(Path(f"/app/temp_data/"))])'''
                rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:anthropause/data/composite_allmet_segment/fua_{poly_id}_{tile}_segment_v2_obs_recomp.csv", str(Path(f"/app/temp_data/"))])
                                                    
                #quality
                existing_qa_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:anthropause/test3/disruption/daily_qa_flags/qa_flag_{poly_id}_{tile}.csv"])
                '''rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:zarrs/visual/composite/fua_{poly_id}_{tile}_flag_qa_stack_v3.csv", str(Path(f"/app/temp_data/"))])'''
                rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:anthropause/data/composite/fua_{poly_id}_{tile}_flag_qa_stack_v3.csv", str(Path(f"/app/temp_data/"))])
                                                    
                #uncertainty
                existing_unc_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:anthropause/test3/disruption/city_uncertainty/unc_{poly_id}_{tile}.txt"])
                '''rclone.with_config(cfg).run_cmd(command="copy",
                                                 extra_args=[f"ceph:zarrs/visual/composite_allmet_Output/fua_{poly_id}_{tile}_U_Output.txt", str(Path(f"/app/temp_data/"))])'''
                rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:anthropause/data/composite_allmet_Output/fua_{poly_id}_{tile}_U_Output.txt", str(Path(f"/app/temp_data/"))])
                       
                #recovery
                existing_rec_files_ls = rclone.with_config(cfg).run_cmd(command="ls", extra_args=[f"ceph:anthropause/test3/recovery/recovery_{poly_id}_{tile}.csv"])
                '''rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:zarrs/visual/composite_stl/fua_{poly_id}_{tile}_rec_stl.csv", str(Path(f"/app/temp_data/"))])'''
                rclone.with_config(cfg).run_cmd(command="copy",
                                                extra_args=[f"ceph:anthropause/data/composite_stl/fua_{poly_id}_{tile}_rec_stl.csv", str(Path(f"/app/temp_data/"))])
                                                    
                                                
                if len(existing_rec_files_ls['out'].decode("utf-8")) >0:
                    print(f"predictions of fua{poly_id} in tile{tile} exists. Skipping")
                else:
                    print(f"forecast of fua{poly_id} in tile{tile} does not exist. Processing")
                   
               
                    ts=pd.read_csv(f"/app/temp_data/fua_{poly_id}_{tile}_time_step_v2_recomp.csv")
                    seg=pd.read_csv(f"/app/temp_data/fua_{poly_id}_{tile}_segment_v2_obs_recomp.csv")
                    qa=pd.read_csv(f"/app/temp_data/fua_{poly_id}_{tile}_flag_qa_stack_v3.csv")
                    unc=pd.read_csv(f"/app/temp_data/fua_{poly_id}_{tile}_U_Output.txt")
                    rec=pd.read_csv(f"/app/temp_data/fua_{poly_id}_{tile}_rec_stl.csv")
                
                    #ts step metrics:
                    column_names = ['dates','pred_ntl','obs_ntl','magnitude','decision','direction','confidence']
                    df_time_step = pd.DataFrame(columns=column_names)
                    df_time_step['dates']=ts.iloc[:,0]
                    df_time_step['pred_ntl']=ts.iloc[:,1]
                    df_time_step['obs_ntl']=ts.iloc[:,2]
                    df_time_step['magnitude']=np.abs(df_time_step.iloc[:,1]-df_time_step.iloc[:,2])
                    df_time_step['decision']=ts.iloc[:,6]
                    df_time_step['direction']=np.where((ts.iloc[:,6]>0), (np.sign(ts.iloc[:,2]-ts.iloc[:,1])),0)
                    df_time_step['confidence']=ts.iloc[:,10]
                    df_time_step.to_csv(str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_time_step_change_metrics.csv")),index=False,float_format='%10.6f')#w_dir_dc
                    rclone.with_config(cfg).run_cmd(command="copy", 
                                                    extra_args=[str(Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_time_step_change_metrics.csv")),
                                                                f"{w_dir_dc}/"])
            
            
                    #seg metrics:
                    column_names_seg=['start','end','inflection','start_rate','end_rate','total_severity','average_severity']
                    df_seg = pd.DataFrame(columns=column_names_seg)
                    df_seg['start']=seg.iloc[:,0]
                    df_seg['end']=seg.iloc[:,1]
                    df_seg['inflection']=seg.iloc[:,2]
                    df_seg['start_rate']=seg.iloc[:,8]
                    df_seg['end_rate']=seg.iloc[:,9]
                    df_seg['total_severity']=seg.iloc[:,5]
                    df_seg['average_severity']=seg.iloc[:,7]
                    df_seg.to_csv(str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_segment_change_metrics.csv")),index=False,float_format='%10.6f')#w_dir_dc
                    rclone.with_config(cfg).run_cmd(command="copy", 
                                                    extra_args=[str(Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_segment_change_metrics.csv")),
                                                                f"{w_dir_cseg}/"])
                
                    #qa metrics
                    column_names_qa=['dates','cloud_free_min','gap_filled_min']
                    df_qa = pd.DataFrame(columns=column_names_qa)
                    df_qa['dates']= qa.iloc[:,0]
                    df_qa['cloud_free_min']=np.abs(qa.iloc[:,1]-100)
                    df_qa['gap_filled_min']=np.abs(qa.iloc[:,3]-100)
                    df_qa.to_csv(str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_quality_metrics.csv")),index=False,float_format='%10.6f')#w_dir_dc
                    rclone.with_config(cfg).run_cmd(command="copy", 
                                                    extra_args=[str(Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_quality_metrics.csv")),
                                                                f"{w_dir_qa}/"])
                    
                    #rec metrics
                    column_names_rec=['dates','ntl_rollingAverage','forecast','difference', 'state']
                    df_rec = pd.DataFrame(columns=column_names_rec)
                    df_rec['dates']= rec.iloc[:,0]
                    df_rec['ntl_rollingAverage']=rec.iloc[:,2]
                    df_rec['forecast']=rec.iloc[:,3]
                    df_rec['difference']=rec.iloc[:,4]
                    df_rec['state']= df_rec['difference'].apply(lambda x: 1 if x<0 else 0) #1 if (rec.iloc[:,2]-rec.iloc[:,3]>0) else 0
                    df_rec.to_csv(str(Path("/app/temp_data",f"fua_{poly_id}_{tile}_recovery_metrics.csv")),index=False,float_format='%10.6f')#w_dir_dc
                    rclone.with_config(cfg).run_cmd(command="copy", 
                                                    extra_args=[str(Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_recovery_metrics.csv")),
                                                               f"{w_dir_rec}/"])
                                                                
                    #unc
                    rclone.with_config(cfg).run_cmd(command="copy", 
                                                    extra_args=[str(Path(f"/app/temp_data",f"fua_{poly_id}_{tile}_U_Output.txt")),
                                                               f"{w_dir_unc}/"])
                                                                
                    #written files                                                 
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_time_step_change_metrics.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_segment_change_metrics.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_quality_metrics.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_recovery_metrics.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_U_Output.txt")))
                    
                   
                    
                    #read files                                                 
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_time_step_v2_recomp.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_segment_v2_obs_recomp.csv")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_flag_qa_stack_v3.csv")))
                    #os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_U_Output.txt")))
                    os.remove(str(Path(f"/app/temp_data", f"fua_{poly_id}_{tile}_rec_stl.csv")))
                

            
            
           
                
            # Update
            print(f"Finished cleanup for {tile} {poly_id}") 
                
        print(f"done")


if __name__ == "__main__":
    
    # Get the system argument for the tile list
    tile_file = argv[1:][0]   
    
    print(f"{tile_file}")
    
    # Call the main function, hard-coding the chosen WSF equator threshold.    
    main(tile_file)
    
     