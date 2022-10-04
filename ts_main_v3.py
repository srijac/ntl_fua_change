import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import math

import tensorflow as tf
print('tf version:', tf.__version__)
import random
random.seed(0)
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from fc_methods_v3 import *
from ts_met_final_v3 import *
from time import time

def get_ra(gf_ntl,observation_count):
    window_size = 30
    
    
    
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
        window_values=gf_ntl[window_start:window_end + 1]
        
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
    
    return ra_ntls

#normalize data from 2017 to 2019 and scale test phase accordingly
def min_max_norm(data,start, tr_ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data[start:tr_ts])
    normalized = scaler.transform(data)
    
    return scaler, normalized

def train_forecast(n,tile,end_d,end_m,end_y,path_obs, path_date,w_dir_wt, w_dir_fc,w_dir_comp):
    
    UA=n
    
    print('CURRENT UA, tile:', UA, tile)
    
    df=np.load(path_obs)
    
    
    date=np.load(path_date)
    
    
    end_day=end_d
    end_month=end_m
    end_year=end_y
    
    print('END YEAR:', end_year)
    
    num_val_gf=df[:,3]
    
    
    
    yr=[]
    month=[]
    day=[]
    for dd in np.arange(0, len(date)):
        yr.append(date[dd].astype(object).year)
        month.append(date[dd].astype(object).month)
        day.append(date[dd].astype(object).day)
    
    yr=np.asarray(yr)
    month=np.asarray(month)
    day=np.asarray(day)
    
    non_nan=[]
    non_nan_yr=[]
    non_nan_month=[]
    non_nan_day=[]
    #print(len(num_val_30))
    print('year',yr)
    num_val_30=num_val_gf
    print('num_val_30',num_val_gf[0] )
    for i in np.arange(0,len(num_val_30)):
        if (~np.isnan(num_val_30[i])):
            non_nan.append(num_val_30[i])
            non_nan_yr.append(yr[i])
            non_nan_month.append(month[i])
            non_nan_day.append(day[i])
    
    
    non_nan=np.asarray(non_nan)
    non_nan_yr=np.asarray(non_nan_yr)
    
    print(yr)
    
    start=np.asarray(np.where(non_nan_yr==2012))#changed from yr to non-nan-yr
    
    print('start array:', start)
    print('start',start[0,0])
    
    print('end yr is:',int(end_year))
    end=np.asarray(np.where(non_nan_yr==int(end_year)))
    
    print('end array:', end)
    print('end:', end[0,-1])
    
    
    start_idx=start[0,0]
    end_idx=end[0,-1]
    print('UA, end year, end', UA, end_year, end_idx, non_nan[end_idx])
    
    num_val=non_nan
    num_val=np.reshape(num_val,(non_nan.shape[0],1))
    init_ntl=num_val
    print('num_val shape:', num_val.shape)
    ra_ntls=get_ra(num_val,num_val.shape[0])
    num_val=ra_ntls
    num_val=np.reshape(num_val,(non_nan.shape[0],1))
    
    mm_obj,norm_ts_mm=min_max_norm(num_val,start_idx,end_idx)# returns the entirely normalized ts, based on parameters 2017: 2019 (training);; UPDATE TO INDEX
    
    inversed = mm_obj.inverse_transform(norm_ts_mm)
    
    #multistep splitting; all methods are given same forecast horizon and input window
    win_l=60
    pred_l=1
    multi_pred_l=30
    
    X_m,y_m=split_multi_step(norm_ts_mm,win_l,multi_pred_l,start_idx,len(norm_ts_mm)) # 1800 corresponds to 2017-01-01 - UPDATE TO INDEX
    print(norm_ts_mm.shape)
    
    
    X_m=X_m.reshape((X_m.shape[0],X_m.shape[1],1))
    y_m=y_m.reshape((y_m.shape[0],y_m.shape[1]))
    
    #CREATING TRAINING AND VALIDATION SPLIT FOR EACH CITY
    X_m_tr=X_m[0:(end_idx-start_idx+1),:,:] # 1005: ~3yrs of training, with a 90 day window; UPDATE TO INDEX, CHECK LENGTH (1005 vs 1095)
    y_m_tr=y_m[0:(end_idx-start_idx+1),:]
    split_idx=random.sample(range(X_m_tr.shape[0]), X_m_tr.shape[0])
    tr_frac=0.8
    val_frac=1-tr_frac
    
    train_idx=split_idx[0:int(np.floor(tr_frac*X_m_tr.shape[0]))]
    val_idx=split_idx[int(np.floor(tr_frac*X_m_tr.shape[0])):len(split_idx)]
    
    train_inp=np.zeros((len(train_idx), X_m_tr.shape[1], X_m_tr.shape[2]))
    train_op=np.zeros((len(train_idx), y_m_tr.shape[1]))
    for idx, value in enumerate(train_idx):
        train_inp[idx,:,:]=X_m_tr[value,:,:]
        train_op[idx,:]=y_m_tr[value,:]
    
    val_inp=np.zeros((len(val_idx), X_m_tr.shape[1], X_m_tr.shape[2]))
    val_op=np.zeros((len(val_idx), y_m_tr.shape[1]))
    
    for idx, value in enumerate(val_idx):
        val_inp[idx,:,:]=X_m_tr[value,:,:]
        val_op[idx,:]=y_m_tr[value,:]


        
    #CALL FORECAST METHDOS
    print('calling CNN')
    fc_cnn(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from CNN prediction')
    print('calling ANN')
    fc_ann(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from ANN prediction')
    print('calling LSTM')
    fc_lstm_tf(norm_ts_mm,X_m,y_m,train_inp,train_op,val_inp, val_op,X_m_tr,y_m_tr,UA,tile, w_dir_wt, w_dir_fc,w_dir_comp)
    print('returned from LSTM prediction')
    print('predictions completed on', UA)
    
    
    


def file_reader(dir_path):
    # get list of city names in dir_path
    f=dir_path
    file_seq_test_n=[]
    for root,subdir,files_pos in os.walk(f,topdown=False):
        print('files/UAs are:',files_pos)
    for files in sorted(files_pos):
        if not files.startswith('.'):
            #print('seq test appending:',files)
            file_seq_test_n.append(files)
    
    return file_seq_test_n
    
    


def forecast_city_list(poly_id, tile,end_d,end_m,end_y,path_obs, path_date,w_dir_wt, w_dir_fc,w_dir_comp, eval_dates):
    print('calling forecast methods on:', poly_id)
    s=time()
    train_forecast(poly_id, tile,end_d,end_m,end_y, path_obs,path_date, w_dir_wt, w_dir_fc,w_dir_comp)
    print('TOOK:', time()-s)
    print('done training/ predicting')
    print('--------------------------')
    print('computing metrics')
    #for n in names:
    compute_metrics(poly_id, tile,end_d,end_m,end_y, path_obs,path_date, eval_dates)