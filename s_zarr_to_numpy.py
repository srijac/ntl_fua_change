from pathlib import Path
import zarr
import numpy as np
from datetime import datetime, timedelta
from random import randint
from c_VNP46A import get_pixel_area
#import matplotlib.pyplot as plt
import csv
from scipy import stats
import statistics
from statistics import mode
from collections import Counter



def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
    
def zarr_to_numpy(zarr_path,poly_zarr):
    # Open the zarr file in read mode
    #poly_zarr = zarr.open(zarr_path, mode='r')

    # Print the structure of the zarr along with its dataset names.
    # This will also display the dimensions, and datatypes of each dataset.
    #print(poly_zarr.tree())
    
    # Iterate over the dataset names as dictionary keys
    for ds_key in poly_zarr.keys():
        # Print each key, as well as the same information (shape and datatype)
        print(ds_key, np.shape(poly_zarr[ds_key]), np.dtype(poly_zarr[ds_key]))
        
    # Get the tile and polygon name (just stripping down the path, nothing to learn here!)
    split_path = str(zarr_path).split('\\')
    split_name = split_path[-1].split('_')
    poly_id = split_name[1]
    tile_name = split_name[-1].strip('.zarr')

    # Get the date range, number of observations, and total number of observations possible.
    start_date = poly_zarr["Dates"][0]
    end_date = poly_zarr["Dates"][-1]
    possible_observations = (start_date - end_date)
    observation_count = len(poly_zarr["Dates"])
    
    # Count of the number of pixels in the FUA polygon (in this tile)
    fua_pixels = np.shape(poly_zarr["DNB_BRDF-Corrected_NTL"])[1]

    # Print a summary of information about the zarr derived from properties of the file

    print('RETRIEVE TIME_SERIES')
    # Get the Night Time Lights
    ntl = poly_zarr["DNB_BRDF-Corrected_NTL"]
    

    # Get the Gap-Filled Night Time Lights
    gapfilled_ntl = poly_zarr["Gap_Filled_DNB_BRDF-Corrected_NTL"]

    q_flag = poly_zarr["Mandatory_Quality_Flag"]

    # Get the Night Time Lights, but replace the fill values (65535) with numpy NaNs
    #filtered_ntl = np.where(ntl != 65535, ntl, np.nan)
    filtered_ntl = np.where((np.array(ntl) < 65535), ntl, np.nan)
    
    gf_filtered=np.where((np.array(gapfilled_ntl)<65535), gapfilled_ntl,np.nan)
    filtered_ntl=filtered_ntl*0.1
    gf_filtered=gf_filtered*0.1
    gf_filtered_avg=np.nanmean(gf_filtered,axis=1)
    ntl_filtered_avg=np.nanmean(filtered_ntl,axis=1)
    
    # Getting the pixel's row globally (0 is top of the world, 21,600 is equator, 43,199 is the bottom of the world)
    tile_v = int(tile_name.split('v')[-1])
    
    print('-----CALLING PIXEL AREA  SCRIPt---------')
    pixel_area_list=np.zeros((fua_pixels,1))
    for r in np.arange(0,fua_pixels):
        pixel_row = poly_zarr["Pixel V"][r]
        global_y = (tile_v * 2400) + pixel_row
        pixel_area_list[r,0] = get_pixel_area(global_y)
    print('-----RETURNED FROM PIXEL AREA  SCRIPt---------')
    wt_avg_at_t=np.zeros((gf_filtered.shape[0],1))
    wt_avg_at_t_wt=np.zeros((gf_filtered.shape[0],1))
    wt_ntl_avg_at_t_wt=np.zeros((gf_filtered.shape[0],1))

    gf_avg_flag=np.zeros((observation_count,1))
    ntl_avg_flag=np.zeros((observation_count,1))
    
    print('-----STARTING WEIGHTED AVG LOOP---------')

    for t in np.arange(0,gf_filtered.shape[0]):#gf_filtered.shape[0]
        #np.multiply(gf_filtered[t,:],pixel_area_list[:])
        temp=0
        temp_ntl=0
        non_nan_p_gf=0
        non_nan_p_ntl=0
        area_sum_gf=0
        area_sum_ntl=0
        gf_flag=[]
        ntl_flag=[]
        print('t is:', t)
        for p in np.arange(0,gf_filtered.shape[1]):#gf_filtered.shape[1]
            #print('t is, p is:', t,p)
            #print('gf_filtered:',gf_filtered[t,p])
            #print('pixel area list:',pixel_area_list[p,0])
            if not(np.isnan(gf_filtered[t,p])):
                temp+=gf_filtered[t,p]*pixel_area_list[p,0]
                non_nan_p_gf+=1
                area_sum_gf+=pixel_area_list[p,0]
                flag=q_flag[t,p]
                gf_flag.append(flag)
            if not(np.isnan(filtered_ntl[t,p])):
                temp_ntl+=filtered_ntl[t,p]*pixel_area_list[p,0]
                non_nan_p_ntl+=1
                area_sum_ntl+=pixel_area_list[p,0]
                ntl_flag.append(flag)
        if non_nan_p_gf==0:
            #wt_avg_at_t[t,0]=np.nan
            wt_avg_at_t_wt[t,0]=np.nan
            gf_avg_flag[t,0]=np.nan
            #ntl_avg_flag[t,0]=most_frequent(ntl_flag)
            #wt_ntl_avg_at_t_wt[t,0]=np.nan
        else:
            #wt_avg_at_t[t,0]=temp/non_nan_p
            wt_avg_at_t_wt[t,0]=temp/area_sum_gf
            #gf_avg_flag[t,0]=stats.mode(gf_flag)[0]
            #gf_avg_flag[t,0]=mode(gf_flag)
            gf_avg_flag[t,0]=most_frequent(gf_flag)
            #wt_ntl_avg_at_t_wt[t,0]=temp_ntl/area_sum
        if non_nan_p_ntl==0:
            wt_ntl_avg_at_t_wt[t,0]=np.nan
            ntl_avg_flag[t,0]=np.nan
        else:
            wt_ntl_avg_at_t_wt[t,0]=temp_ntl/area_sum_ntl
            ntl_avg_flag[t,0]=most_frequent(ntl_flag)
            

    
    print('ts flag:', gf_avg_flag.shape, ntl_avg_flag.shape)
            
    ts_stack=np.zeros((observation_count,6))#ntl,ntl-filtered,non_weighted_avg, wt_ntl_filtered, gap-filled, gap-filled-filtered,non_weighted_gap_ wt-gap-filled, flags
    #ts_stack[:,0]=ntl[]
    #ts_stack[:,1]=filterd_ntl
    ts_stack[:,0]=ntl_filtered_avg
    ts_stack[:,1]=wt_ntl_avg_at_t_wt[:,0]
    #ts_stack[:,4]=gapfilled_ntl
    #ts_stack[:,5]=gf_filtered
    ts_stack[:,2]=gf_filtered_avg
    ts_stack[:,3]=wt_avg_at_t_wt[:,0]
    ts_stack[:,4]=gf_avg_flag[:,0]
    ts_stack[:,5]=ntl_avg_flag[:,0]
    #ts_stack[:,6]=np.datetime64(poly_zarr["Dates"][0:])
    

    
    window_size = 6
    
   
    
    ra_ntls=[]
    
    print('obs count', (observation_count))
        
    for t in np.arange(0,observation_count):
        #print('ntl at t:', wt_avg_at_t_wt[t])
        if t-int(window_size/2)<0:
            window_start=0
        else:
            window_start=t-int(window_size/2)
        if (int(window_size/2)+t)>=(observation_count):
            window_end=(observation_count-1)
        else:
            window_end=(int(window_size/2)+t)
        #print('t', t, wt_avg_at_t_wt[window_start:window_end + 1])
        window_values=wt_avg_at_t_wt[window_start:window_end + 1]
        
        # If the window is nothing but nans
        if np.sum(np.isnan(window_values)) == len(window_values):
            # Fill the rolling average value
            ra_ntls.append(np.nan)
        # Otherwise (at least one observation)
        else:
            # Append the mean (ignoring nans)
            ra_ntls.append(np.nanmean(window_values))
        #print('ra at t:', ra_ntls)
    
    ra_ntls=np.asarray(ra_ntls)
    print('ra ntl shape',ra_ntls.shape)
    
    return ts_stack, poly_zarr["Dates"][0:]
    
    
    