# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:59:07 2021

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
import numpy.ma as ma
import math
import pandas as pd

lat_start = -75
lat_end = -65
lon_start = -142
lon_end = -85

#take the x,y grid corners from lat,lon grid corners
#x_downdown, y_downdown =  cvert_latlontoxy(lat_downdown,lon_downdown)
lat_num = 30
lon_num = 20
lat_step = (lat_end-lat_start)/lat_num
lon_step = (lon_end-lon_start)/lon_num
lat_array = np.linspace(lat_start,lat_end,lat_num)
lon_array = np.linspace(lon_start,lon_end,lon_num)
lat_mesh,lon_mesh = np.meshgrid(lat_array,lon_array)

lonfile = open("Data\\pss25lons_v3.dat", "rb")
latfile = open("Data\\pss25lats_v3.dat", "rb")
lon = np.fromfile(lonfile, dtype=np.int32)
lat = np.fromfile(latfile, dtype=np.int32)
lon = lon/100000
lat = lat/100000

lat = lat.reshape(332, 316)
lon = lon.reshape(332, 316)

def concentration_monthly(year,month):
    ice_con_array = [] 
    
    #save_whole = str('Results\\xyandlatlonGrid_thick_icesat2_' + year + month + '.npz')
    #dict_xyGrid_thick_icesat2 = np.load('D:\ATL10_2018to2021\\' + save_whole)
    #x_icesat2 = dict_xyGrid_thick_icesat2['x_icesat2_svd']
    #y_icesat2 = dict_xyGrid_thick_icesat2['y_icesat2_svd']
    day_count = 0
    file_format = "Data\\bt_"+year+month+"*_f17_v3.1_s.bin"
    for FILE_NAME in glob.iglob(file_format):
        print((FILE_NAME))
        #f = open("bt_20201031_f17_v3.1_s.bin", "rb")
        ice = np.fromfile(FILE_NAME, dtype=np.uint16)
        print(ice.shape)    
        ice = ice.reshape(332, 316)
        ice = ice/10
        ice = ma.masked_greater(ice, 100)
        ice_con_array += [ice]
        day_count += 1
        print(np.max(ice))
    ice_con_array_3D = np.asarray(ice_con_array)
    
    #np.savez_compressed('Results/ice_con_Oct2020.npz', ice_con_array_svd = ice_con_array_3D)
    #dict_ice_con_array = np.load('D:\IceThickness_LinearAssumption_Monthly\\Results\\ice_con_Oct2020.npz')
    #ice_con_array_3D = dict_ice_con_array['ice_con_array_svd']
    #ice_con_array_3D = np.asarray(ice_con_array_3D)
    #print(ice_con_array.shape)
    ice_con_avg_2D = np.zeros((332,316))
     
    for i in range(332):
        for j in range(316):
            ele_arr = [ice_con_array_3D[k][i][j] for k in range(day_count)]
            ele_arr = np.asarray(ele_arr)
            ele_arr = ma.masked_greater(ele_arr, 100)
            ice_con_avg_2D[i][j] = np.mean(ele_arr)
            #print(np.nanmax(ele_arr)) 
    print(ice_con_avg_2D.shape)
    print(np.nanmax(ice_con_avg_2D))        
    print(np.nanmean(ice_con_avg_2D))
    print(np.nanmedian(ice_con_avg_2D))
    print(np.nanmin(ice_con_avg_2D))
    ice_con_avg_2D = np.asarray(ice_con_avg_2D)
          
    further_avg_ice_con = np.zeros((lon_num,lat_num))
    for i in range(lat_num):
        for j in range(lon_num):
            lat_start_sub = lat_start + lat_step*i
            lat_end_sub = lat_start + lat_step*(i+1)
            lon_start_sub = lon_start + lon_step*j
            lon_end_sub = lon_start + lon_step*(j+1)
            #print(x_start_sub)
            #print(x_end_sub)
            #print(y_start_sub)
            #print(y_end_sub)
            further_avg_ice_con[j][i] = np.nanmean(np.where((lat > lat_start_sub) & (lat < lat_end_sub) & (lon > lon_start_sub) & (lon < lon_end_sub), ice_con_avg_2D, float("NaN")))
            print('Avg Ice Concentration is ',further_avg_ice_con[j][i]) 
            #time_loop = datetime.datetime.now()
            #print(time_loop)
    
    print(np.asarray(further_avg_ice_con).shape)
    
    #Save and load plotting data for the relevant region (Amundsen) arrays locally
    
    np.savetxt('Results/latlon_further_avg_ice_con' + year + month + '.csv', further_avg_ice_con, delimiter=',')
    further_avg_ice_con = np.loadtxt('Results/latlon_further_avg_ice_con' + year + month + '.csv', delimiter=',')
    further_avg_ice_con = np.asarray(further_avg_ice_con)
    
    for i in range(len(further_avg_ice_con)):
        for j in range(len(further_avg_ice_con[i])):
            if pd.isna(further_avg_ice_con[i][j]):
                further_avg_ice_con[i][j] = 120
    
    #Plot ice concentration
    plt.figure()
    #plt.pcolormesh(x_mesh,y_mesh,further_avg_ice_con)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.title('Ice concentration [%] in the Amundsen Sea region in ' + month + '/' + year)
    #plt.colorbar()
    plt.contour(lat_mesh,lon_mesh,further_avg_ice_con,levels=[15,119],colors=['k','r'])
    plt.savefig(str('Results/latlon_contour_Amundsen_xyGrid_IceConcentration' + year + month + '.pdf'))
    plt.savefig(str('Results/latlon_contour_Amundsen_xyGrid_IceConcentration' + year + month + '.png'))
    
    #plt.figure()
    #plt.pcolormesh(lat_mesh,lon_mesh,further_avg_ice_con)
    #plt.colorbar()
    
concentration_monthly('2018','10')
concentration_monthly('2018','11')
concentration_monthly('2018','12')
concentration_monthly('2019','01')
concentration_monthly('2019','02')
concentration_monthly('2019','03')
concentration_monthly('2019','04')
concentration_monthly('2019','05')
concentration_monthly('2019','06')
concentration_monthly('2019','07')
concentration_monthly('2019','08')
concentration_monthly('2019','09')
concentration_monthly('2019','10')
concentration_monthly('2019','11')
concentration_monthly('2019','12')
concentration_monthly('2020','01')
concentration_monthly('2020','02')
concentration_monthly('2020','03')
concentration_monthly('2020','04')
concentration_monthly('2020','05')
concentration_monthly('2020','06')
concentration_monthly('2020','07')
concentration_monthly('2020','08')
concentration_monthly('2020','09')
concentration_monthly('2020','10')
concentration_monthly('2020','11')
concentration_monthly('2020','12')
