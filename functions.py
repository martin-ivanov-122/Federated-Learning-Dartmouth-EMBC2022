    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:52:06 2020

@author: binhnguyen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import *
import csv
from sklearn.cluster import KMeans
from skimage.restoration import denoise_wavelet 
import pywt
import time
from math import radians, cos, sin, asin, sqrt


# Reading GPS time value
def GPS_csv_reader (subject, sensor_num, col, decision):
    sensor = 'activity', 'audio', 'bluetooth','conversation', 'dark','gps','phonecharge','phonelock','wifi','wifi_location'
    dir_path = '../'
    user = "_" + subject+'.csv'
    name = 'sensing/'+ sensor[sensor_num] + "/" + sensor[sensor_num]
    full_path = dir_path + name + user
    
    output = []
    with open(full_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (decision == 1):
                output.append(float(row[col]))
            else:
                output.append(row[col])
    file.close()
    
    return (np.asarray (output))



# Reads the data from sensor folder
def sensor_val (subject, sensor_num):
    
    sensor = 'activity', 'audio', 'bluetooth','conversation', 'dark','gps','phonecharge','phonelock','wifi','wifi_location'
    dir_path = '../'
    
    if (sensor_num==2):
        name = 'sensing/'+ sensor[sensor_num] + "/bt"    
        
    else:
        name = 'sensing/'+ sensor[sensor_num] + "/" + sensor[sensor_num]
    
    user = "_" + subject+'.csv'
    full_path = dir_path + name + user
    
    # print ("Reading from file %s" %full_path)
    df = pd.read_csv(full_path)
    return df


# Reads the data from EMA folder
def ema_val (subject, ema_num):
    
    sensor = "Activity","Administration's response","Behavior","Exercise",	"Lab","Mood","Mood 1"	,"Mood 2",	"Sleep"	,"Social",	"Stress"
    
    dir_path = '../'
    
    name = '/EMA/response/'+ sensor[ema_num] + "/" + sensor[ema_num]

    
    user = '_u'+subject+'.json'
    full_path = dir_path + name + user
    
    df = pd.read_json(full_path)
    return df


# Changes the survey to INT values
def survey_encoder (response):
    count = 0
    output = [0]*len(response)
    
    for resp in response: 
        if (resp == 'Disagree Strongly'):
            output[count] = -2
        elif (resp == 'Disagree a little'):
            output[count] = -1
        elif (resp == 'Neither agree nor disagree'):
            output[count] = 0
        elif (resp == 'Agree a little'):
            output[count] = 1
        elif (resp == 'Agree strongly'):
            output[count] = 2
        else:
            output[count] = 100
        count+=1
        
    return output
    
# Shit analysis
def activity_ (df):
    count = 0
    timestamp = df.iloc[:,0]
    sig = df.iloc [:,1]
    
    sig_max = np.max(timestamp)
    sig_min = np.min(timestamp)
    delta = sig_max-sig_min
    sig_new = np.ones (delta) 
    
    for i in range (0,delta-1):
        
        if (timestamp [count] == timestamp[count+1]):
            count+=1
        
        if (i+sig_min == timestamp [count]):
            sig_new [i] = sig[count]
            count+=1
        else:
            sig_new [i] = sig_new[i-1]
    return sig_new,count



# Analysis for Activity from passive sensors
def activity_passive (subject, sensor_num):
    for i in subject:    
        try:
            activity = sensor_val (str(i), sensor_num) 
            # activity = sensor_val (subject, sensor_num) 
            activity_col = activity.iloc[:,1]
            
            if (np.mean(activity_col) > 0.01):
                print ("%s is good" %i)
            else: 
                print ("%s isn't above average but is %f" %(i,np.mean(activity_col)))
                
               
        except:
            print ("%s does not exist" %i)
            
            
 # Analysis for Stress from EMA sensors           
def stress_active(subject, ema_num):
    final_date =[]
    
    def null_conversion(null_col):    
        for i in range (len(null_col)):
            try:
                if (isnan(null_col[i])):
                    null_col[i] = 0
                    
            except:
                if (len(null_col[i])>1):
                    null_col[i] = 0
                else:
                    null_col[i] = int(null_col[i])
        return null_col
    

    
    i = subject
    try:
        # ema = ema_val ('0'+str(i), ema_num)     
        ema = ema_val (str(i), ema_num)     
            
        # Clean level col
        level_col = (ema['level'])    
        where_are_NaNs = isnan(level_col)
        level_col[where_are_NaNs] = 0    
        
        # Clean null col
        null_col = (ema['null'])    
        null_col = null_conversion(null_col)
        
        final_stress = null_col + level_col
        
        final_date.append (ema['resp_time'])
        
    except:
        final_stress, final_date = 1000,1000

    return (final_stress, final_date)


# Put PHQ9 survey into DF
def survey_reader (surv_num):
    
    survey = 'PHQ-9.csv','panas.csv'
    dir_path = '../'
    name = 'survey/'+ survey[surv_num] 
    
    full_path = dir_path + name

    df = pd.read_csv(full_path)
    return df


# Changes the survey to INT values
def survey_encoder_phq (response):
    count = 0
    output = [0]*len(response)
    
    for resp in response: 
        if (resp == 'Not at all'):
            output[count] = 0
        elif (resp == 'Several days'):
            output[count] = 1
        elif (resp == 'More than half the days'):
            output[count] = 2
        elif (resp == 'Nearly every day'):
            output[count] = 3
        else:
            output[count] = 100
        count+=1
        
    return output



# Scores the severity of the PHQ
def phq_severity (phq):
    output = [] 
    for i in phq:
        if (i <=4):
            output.append ("Normal")
        elif (i>=5 and i<=9):
            output.append ("Mild")
        elif (i>=10 and i<=14):
            output.append ("Moderate")
        elif (i>=15 and i<=19):
            output.append ("Moderate to severe")
        elif (i>=20 and i<=27):
            output.append ("Severe")
        else:
            print ("Unvalid")
    return output



# Scores the severity of the PHQ where labels are Normal and Other
def phq_severity_sri (phq):
    output = [] 
    for i in phq:
        if (i <=4): #originally 4
            output.append ("Normal")
        else:
            output.append ("Mild to severe")
    return output


# Scores the severity of the PHQ where labels are Normal and Other
def phq_severity_binh (phq):
    output = [] 
    for i in phq:
        if (i <=2):
            output.append ("Normal")
        elif (i > 2 and i <= 7):
            output.append ("Mild to moderate")
        else:
            output.append ("Moderate to severe")
                           
    return output


# Fct to take the difference in time of a file (e.g phone charge)
def time_difference (val):
    return (np.var (val.iloc[:,1]-val.iloc[:,0]), \
            np.mean (val.iloc[:,1]-val.iloc[:,0]),\
               np.sum (val.iloc[:,1]-val.iloc[:,0]))


# Fct to find the ratio of PA/NA and see if it's greater than 1 or not
def panas_label_seperater (val):
    output = np.zeros (len(val))
    output[val >= 1] = 1
    return output





# K-means algorithm as indicited by paper
def kmeans_cluster (gps_time, gps_lat, gps_lon, decision):

    X= np.transpose (np.array([gps_lat,gps_lon]))
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, \
                        random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    wcss = normalization1 (wcss)

    der_wcss = (diff(wcss))

    cutoff = -0.05
    for i in range (len(wcss) - 3):
        if (der_wcss [i]> cutoff):
            n_clust = i
            break
        
    kmeans = KMeans(n_clusters=n_clust, init='k-means++', max_iter=300, \
                    n_init=10,random_state=0)
    kmeans.fit(X)
    pred_y = kmeans.predict(X)

    # Testing purposes

    # print ("Number of cluster is %d" %n_clust) 
    # print (len(X[:,0][X[:,0]<40])) 
    # Proof that majority of 
    # sample points above lat =4
    
    # End of testing
    
    # Plotting dec 1
    if (decision == 1):
        
        # Red centroids
        plt.figure ()
        plt.subplot(211)
        plt.plot(diff(wcss))
        plt.title ("Derivative of WCSS")
        plt.xlabel ("Cluster")
        plt.ylabel ("Distance")
        plt.subplot(212)
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.title ("Original data with centroid circles")
        plt.xlabel ("Latitude")
        plt.ylabel ("Longitutde")
        
    # Plotting dec 2
    elif (decision ==2):
        x_spot = np.arange (0,10)
        plt.figure ()
        plt.scatter (x_spot,wcss)
        plt.title ("WCSS")
        plt.xlabel ("Cluster")
        plt.ylabel ("Distance")
        plt.title ("Number of cluster is %d" %n_clust)
        
        
        
    
    # Time in clusters - GPS points were collected every 10 minutes
    # https://studentlife.cs.dartmouth.edu/dataset.html#sec:data_dir:ema_dir
    c = []
    for i in range (n_clust):
        c.append (gps_time [pred_y == i])
    
    # Normalization 
    # Total time during each cluters in seconds/Total time in study in seconds
    normalization = (gps_time[len(gps_time)-1]-gps_time[0])
    # normalization = len(gps_time) * 60 * 10
        
    # Get all clusters
    c_time = []
    for i in range (n_clust):
        c_time.append (len(c[i]) * 60 * 10 / normalization)
        
    # Get top 3 clusters
    if (n_clust >=1):
        c1 = (len(c[0]) * 60 * 10 / normalization)
        c2=0
        c3=0
    if (n_clust >=2):
        c2 = (len(c[1]) * 60 * 10 / normalization)
    if (n_clust >=3):
        c3 = (len(c[2]) * 60 * 10 / normalization)

    return (c1,c2,c3,c_time, n_clust)


def normalization1 (x):
    a=min(x)
    b=max(x)
    normalized = np.zeros(len(x))
    for i in range(len(x)):
        normalized[i] = (x[i]-a)/(b-a)
    return (normalized)



def entropy_fct (n_clust,c):
    
    entropy = 0 
    
    for i in range (len(c)):
        entropy -= (c[i]) * log (c[i])
        
    if (n_clust == 1):
        normalized_entropy = 0
    else:
        normalized_entropy = entropy / log(n_clust)
    
    return entropy, normalized_entropy 
    

def percentage_features (gps_lat,gps_lon,gps_time,decision,subject):    
    
    # Get time local to EST from 12-6 am
    home = []
    # time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(gps_time[2044]))
    for i in range (len(gps_time)):
        s = time.strftime('%H:%M', time.localtime(gps_time[i]))
        hour = int (s[0:2])
        
        if ((hour>=0) and (hour<6)):
            home.append (i)

    night_lat = gps_lat.iloc[home]
    night_lon = gps_lon.iloc[home]
    night_lat_red = night_lat[night_lat.between(night_lat.quantile(.10),night_lat.quantile(.90))]
    night_lon_red = night_lon[night_lat.between(night_lat.quantile(.10),night_lat.quantile(.90))]
    
    # # Plotting purposes
    # plt.scatter (night_lat,night_lon)
    # plt.scatter (night_lat_red,night_lon_red)
    
    
    # Get the kmeans of the data to see where the centroid is for the Home
    kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, \
            n_init=10,random_state=0)
    X = np.transpose (np.array([night_lat_red,night_lon_red]))
    # X = np.transpose (np.array([night_lat,night_lon]))
    kmeans.fit(X)
    lat_centroid, lon_centroid = kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]
    

    
    # Compare all points to see where home is 
    point = (np.array([gps_lat,gps_lon]))
    centroid = (np.array([lat_centroid,lon_centroid]))
    dist = dist_c2p(centroid, point)
    
    # Home coordinates
    home_lat =  point [0][dist<np.mean(dist)]
    home_lon =  point [1][dist<np.mean(dist)]
    
    if (decision == 1):
        # Visualize the home
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(gps_lat, gps_lon)
        ax1.scatter(home_lat,home_lon,c='red')
        ax1.title.set_text('Red = Home, Blue = GPS')
    
    home_d = len(home_lat)/len(gps_lat)
    
    # For testing purposes
    # print (len(gps_lat [gps_lat <44.5]))
    
    
    # GPS moving percentage
    gps_moving = GPS_csv_reader(subject, 5, 'travelstate',0)
    move_p = len(gps_moving[gps_moving == 'moving'])/len(gps_lat)
    
    return (home_d, move_p)    


# Find the distance from centroid to point
def dist_c2p(centroid, point):
    

    c_x = centroid[0]
    c_y = centroid[1]
    
    p_x = point [0]
    p_y = point [1]
    
    output = np.zeros (len(p_x))
    for i in range (len(p_x)):
        left = c_y-p_y[i]
        right = c_x-p_x[i]
        inner = left**2 + right **2
        output[i] = (np.sqrt (inner))
        
    return output


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def var_trends_convo (convo):
    
    #### Coding to seperate the information to days
    new_c = convo - min(convo['start_timestamp'])
    # Assuming 86400 seconds is a day
    start = new_c['start_timestamp']
    
    c = 0
    samples_per_day1 = 0
    num_days = int(ceil (np.max(new_c['start_timestamp'])/86400))
    c_dur = np.zeros(num_days,dtype=int)

    
    for i in range (1,num_days):
        
        hold1 = start [(start < (86400 * i))]
        hold2 = hold1 [(hold1 > (86400 * (i-1)))]
        
        
        samples_per_day2 = len (hold2)

        focus = new_c[samples_per_day1:samples_per_day1+samples_per_day2]
        x,x,c_dur[c] = (time_difference(focus))
        
        samples_per_day1 = samples_per_day2
        c+=1
    
    
    
    return (c_dur)
    


def var_trends_activity (act):
    
    #### Coding to seperate the information to days
    new_activity = act['timestamp'] - min(act['timestamp'])
    # Assuming 86400 seconds is a day
    
    c = 0
    samples_per_day1 = 0
    num_days = int(ceil (np.max(new_activity)/86400))
    act_s = np.zeros(num_days,dtype=int)
    act_w = np.zeros(num_days,dtype=int)
    act_r = np.zeros(num_days,dtype=int)
    
    for i in range (1,num_days):
        
        hold1 = new_activity [(new_activity < (86400 * i))]
        hold2 = hold1 [(hold1 > (86400 * (i-1)))]
        
        
        samples_per_day2 = len (hold2)

        focus = act [' activity inference'][samples_per_day1:samples_per_day1+samples_per_day2]
        act_s [c] = len (focus [focus == 0])
        act_w [c] = len (focus [focus == 1])
        act_r [c] = len (focus [focus == 2])        
        
        samples_per_day1 = samples_per_day2
        c+=1
    
    
    
    return (np.var (act_s),np.var (act_w),np.var (act_r),act_s,act_w,act_r)



def var_trends_audio (audio):
    
    #### Coding to seperate the information to days
    new_audio = audio['timestamp'] - min(audio['timestamp'])
    # Assuming 86400 seconds is a day
    
    c = 0
    samples_per_day1 = 0
    num_days = int(ceil (np.max(new_audio)/86400))
    aud_s = np.zeros(num_days,dtype=int)
    aud_v = np.zeros(num_days,dtype=int)
    aud_n = np.zeros(num_days,dtype=int)
    
    for i in range (1,num_days):
        
        hold1 = new_audio [(new_audio < (86400 * i))]
        hold2 = hold1 [(hold1 > (86400 * (i-1)))]
        
        
        samples_per_day2 = len (hold2)

        focus = audio [' audio inference'][samples_per_day1:samples_per_day1+samples_per_day2]
        aud_s [c] = len (focus [focus == 0])
        aud_v [c] = len (focus [focus == 1])
        aud_n [c] = len (focus [focus == 2])        
        
        samples_per_day1 = samples_per_day2
        c+=1
    

    return (np.var(aud_s), np.var (aud_v), np.var (aud_n),aud_s,aud_v,aud_n)
    
    


def var_trends_convo (convo):
    
    #### Coding to seperate the information to days
    new_conv = convo['start_timestamp'] - min(convo['start_timestamp'])
    # Assuming 86400 seconds is a day
    
    c = 0
    samples_per_day1 = 0
    num_days = int(ceil (np.max(new_conv)/86400))
    dur = np.zeros(num_days,dtype=int)
    
    for i in range (1,num_days):
        
        hold1 = new_conv [(new_conv < (86400 * i))]
        hold2 = hold1 [(hold1 > (86400 * (i-1)))]
        
        
        samples_per_day2 = len (hold2)

        focus = convo [samples_per_day1:samples_per_day1+samples_per_day2]
        null, null, dur [c] = time_difference (focus)
     
        
        samples_per_day1 = samples_per_day2
        c+=1
    
    
    
    return (dur)
    
def wavelet_denoise (signal, plot):
    # Tutorial taken from the following website:
    # https://www.youtube.com/watch?v=HSG-gVALa84&ab_channel=Dr.AjayKumarVerma

    denoised_sig = denoise_wavelet(signal, method = 'VisuShrink', \
       wavelet = 'haar', wavelet_levels=None, rescale_sigma=True)
    # plt.plot (denoised_sig)
    
    
    if (plot == True):
        # Plotting of graphs
        fig, axs = plt.subplots(2)
        fig.suptitle('Original vs Wavelet denoised')
        axs[0].plot(signal)
        axs[1].plot(denoised_sig*(10**19))
        axs[1].set_ylim([0, max(signal)])
        
    return (denoised_sig*(10**19))
  
def wavelet_denoise2 (signal, plot):
    
    # Tutorial taken from
    # https://cagnazzo.wp.imt.fr/files/2018/04/tp_sd205.pdf
    wavelet = pywt.Wavelet('haar')
    
    # Approximation is the first index
    # Coefficients are the indices after
    coeffs = pywt.wavedec(signal, wavelet)
    
    # Thresholding
    threshold = np.std (signal)
    NewWaveletCoeffs = []
    NewWaveletCoeffs.append (coeffs[0])
    for i in range (1,len(coeffs)):
        NewWaveletCoeffs.append(pywt.threshold(coeffs[i], threshold, 'soft'))
    
    denoised_sig = pywt.waverec(NewWaveletCoeffs, wavelet)
    
    if (plot == True):
        # Plotting of graphs
        fig, axs = plt.subplots(2)
        fig.suptitle('Original vs Wavelet denoised')
        axs[0].plot(signal)
        axs[1].plot(denoised_sig)
        axs[1].set_ylim([0, max(signal)])
        
    return (denoised_sig)
    
def matlab2python (filename, len_survey):
    walk_trend = np.zeros ((len_survey,4))
    r=0
    c=0
    test = []
    f = open(filename, "r")
    for file_line in f:
    
        walk_trend [r,c] = file_line  
    
        r+=1
        
        if (r==len(survey[0])):
          c+=1
          r=0
    
    f.close()
    
    return (walk_trend)

def load_file(filename):
    with open(filename) as f:
        output = f.readlines()
    f.close ()
    
    return (output)
    
    
    