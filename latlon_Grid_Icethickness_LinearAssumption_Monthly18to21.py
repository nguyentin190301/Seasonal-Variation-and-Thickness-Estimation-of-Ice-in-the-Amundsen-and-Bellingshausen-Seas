# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 18:55:10 2021

@author: Acer
"""

import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
import pandas as pd
import cartopy.crs as ccrs

lat_start = -75
lat_end = -65
lon_start = -142
lon_end = -85

#x_start, y_start = cvert_latlontoxy(lat_start,lon_start)
#x_end, y_end = cvert_latlontoxy(lat_end,lon_end)
#print(x_start,y_start,x_end,y_end)

lat_num = 30
lon_num = 20
lat_step = (lat_end-lat_start)/lat_num
lon_step = (lon_end-lon_start)/lon_num
lat_array = np.linspace(lat_start,lat_end,lat_num)
lon_array = np.linspace(lon_start,lon_end,lon_num)
lat_mesh,lon_mesh = np.meshgrid(lat_array,lon_array)
lat_mesh = np.asarray(lat_mesh)
lon_mesh = np.asarray(lon_mesh)
iceberg_thickness_cap = 5*2.79+0.169
print(iceberg_thickness_cap)

def cvert_latlontoxy(latin,lonin):
    slat = 70
    slon = 0
    sgn = -1
    pi = np.pi
    E2 = 0.006693883
    E = np.sqrt(E2)
    R = 6378.273 #radius of earth
    
    sl = math.radians(slat)
    lat = sgn*math.radians(latin)
    lon = math.radians(lonin)
    lon0 = math.radians(slon)
    
    if (abs(sl-pi/2) < 0.001):
        T = ((1-np.sin(lat))/(1+np.sin(lat))*((1+E*np.sin(lat))/(1-E*np.sin(lat)))**E)**0.5
        rho = 2*R*T/((1+E)**(1+E)*(1-E)**(1-E))**0.5
    else:
    	T = np.tan(pi/4-lat/2)/((1-E*np.sin(lat))/(1+E*np.sin(lat)))**(E/2)
    	tc = np.tan(np.pi/4-sl/2)/(((1-E*np.sin(sl))/(1+E*np.sin(sl)))**(E/2))
    	mc = np.cos(sl)/np.sqrt(1-E2*np.sin(sl)**2)
    	rho = R*mc*T/tc
    y = -rho*sgn*np.cos(sgn*(lon-lon0))
    x = rho*sgn*np.sin(sgn*(lon-lon0))
    return x, y

def thickness_linear_monthly(year,month):    
    '''
    latitude_icesat2 = []
    longitude_icesat2 = []
    fb_height = []
    file_format =  "Data\\ATL10-02_" + year + month + "**.h5"
    for FILE_NAME in glob.iglob(file_format):
        try:        
            with h5py.File(str(FILE_NAME), mode='r') as f_icesat2:        
                latvar_1R = f_icesat2['/gt1r/freeboard_beam_segment/beam_freeboard/latitude']
                latvar_1L = f_icesat2['/gt1l/freeboard_beam_segment/beam_freeboard/latitude']
                latvar_2R = f_icesat2['/gt2r/freeboard_beam_segment/beam_freeboard/latitude']
                latvar_2L = f_icesat2['/gt2l/freeboard_beam_segment/beam_freeboard/latitude']
                latvar_3R = f_icesat2['/gt3r/freeboard_beam_segment/beam_freeboard/latitude']
                latvar_3L = f_icesat2['/gt3l/freeboard_beam_segment/beam_freeboard/latitude']
                latitude_icesat2_1R = latvar_1R[:]
                latitude_icesat2_1L = latvar_1L[:]
                latitude_icesat2_2R = latvar_2R[:]
                latitude_icesat2_2L = latvar_2L[:]
                latitude_icesat2_3R = latvar_3R[:]
                latitude_icesat2_3L = latvar_3L[:]
                latitude_icesat2 = np.concatenate((latitude_icesat2, latitude_icesat2_1R, latitude_icesat2_1L, latitude_icesat2_2R, latitude_icesat2_2L, latitude_icesat2_3R, latitude_icesat2_3L))
                  
                lonvar_1R = f_icesat2['/gt1r/freeboard_beam_segment/beam_freeboard/longitude']
                lonvar_1L = f_icesat2['/gt1l/freeboard_beam_segment/beam_freeboard/longitude']
                lonvar_2R = f_icesat2['/gt2r/freeboard_beam_segment/beam_freeboard/longitude']
                lonvar_2L = f_icesat2['/gt2l/freeboard_beam_segment/beam_freeboard/longitude']
                lonvar_3R = f_icesat2['/gt3r/freeboard_beam_segment/beam_freeboard/longitude']
                lonvar_3L = f_icesat2['/gt3l/freeboard_beam_segment/beam_freeboard/longitude']
                longitude_icesat2_1R = lonvar_1R[:]
                longitude_icesat2_1L = lonvar_1L[:]
                longitude_icesat2_2R = lonvar_2R[:]
                longitude_icesat2_2L = lonvar_2L[:]
                longitude_icesat2_3R = lonvar_3R[:]
                longitude_icesat2_3L = lonvar_3L[:]
                longitude_icesat2 = np.concatenate((longitude_icesat2, longitude_icesat2_1R, longitude_icesat2_1L, longitude_icesat2_2R, longitude_icesat2_2L, longitude_icesat2_3R, longitude_icesat2_3L))
                
                fb_height_name_1R = '/gt1r/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                fb_height_name_1L = '/gt1l/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                fb_height_name_2R = '/gt2r/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                fb_height_name_2L = '/gt2l/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                fb_height_name_3R = '/gt3r/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                fb_height_name_3L = '/gt3l/freeboard_beam_segment/beam_freeboard/beam_fb_height'
                
                fb_height_var_1R = f_icesat2[fb_height_name_1R]   
                fb_height_var_1L = f_icesat2[fb_height_name_1L]   
                fb_height_var_2R = f_icesat2[fb_height_name_2R]   
                fb_height_var_2L = f_icesat2[fb_height_name_2L]   
                fb_height_var_3R = f_icesat2[fb_height_name_3R]   
                fb_height_var_3L = f_icesat2[fb_height_name_3L]
                
                fb_height_1R = fb_height_var_1R[:]
                fb_height_1L = fb_height_var_1L[:]
                fb_height_2R = fb_height_var_2R[:]
                fb_height_2L = fb_height_var_2L[:]
                fb_height_3R = fb_height_var_3R[:]
                fb_height_3L = fb_height_var_3L[:]
                
                fb_height = np.concatenate((fb_height, fb_height_1R, fb_height_1L, fb_height_2R,  fb_height_2L, fb_height_3R, fb_height_3L))
        except:
            continue
    fb_height = np.array(fb_height)
    fb_height_dim = fb_height.shape[0]
    for i in range(fb_height_dim):
        #if fb_height[i] > 100:
        if fb_height[i] > 5: #avoid iceberg
            fb_height[i] = float("NaN")
            
    x_icesat2 = [cvert_latlontoxy(latin,lonin)[0] for latin, lonin in zip(latitude_icesat2,longitude_icesat2)]
    y_icesat2 = [cvert_latlontoxy(latin,lonin)[1] for latin, lonin in zip(latitude_icesat2,longitude_icesat2)]
    ice_thickness = [height*2.79+0.169 for height in fb_height]
    #time2 = datetime.datetime.now()
    #interval12 = time2 - time1
    #print(interval12)
    
    x_icesat2 = np.asarray(x_icesat2)
    y_icesat2 = np.asarray(y_icesat2)
    ice_thickness = np.asarray(ice_thickness)
    '''
    '''save_whole = str('Results\\xyandlatlonGrid_thick_icesat2_' + year + month + '.npz')
    #np.savez_compressed(save_whole, latitude_icesat2_svd = latitude_icesat2, longitude_icesat2_svd = longitude_icesat2, x_icesat2_svd=x_icesat2, y_icesat2_svd=y_icesat2, ice_thickness_svd=ice_thickness)
    dict_xyGrid_thick_icesat2 = np.load(save_whole)
    #dict_xyGrid_thick_icesat2 = np.load('D:\ATL10_2018to2021\\' + save_whole)
    lat_icesat2 = dict_xyGrid_thick_icesat2['latitude_icesat2_svd']
    lon_icesat2 = dict_xyGrid_thick_icesat2['longitude_icesat2_svd']
    ice_thickness = dict_xyGrid_thick_icesat2['ice_thickness_svd']
    
    ice_thickness = np.array(ice_thickness)
    ice_thickness_dim = ice_thickness.shape[0]
    for i in range(ice_thickness_dim):
        #if fb_height[i] > 100:
        if ice_thickness[i] > iceberg_thickness_cap: #avoid iceberg
            ice_thickness[i] = float("NaN")    
    
    print(np.nanmax(lat_icesat2))
    print(np.nanmean(lat_icesat2))
    print(np.nanmedian(lat_icesat2))
    print(np.nanmin(lat_icesat2))
    
    print(np.nanmax(lon_icesat2))
    print(np.nanmean(lon_icesat2))
    print(np.nanmedian(lon_icesat2))
    print(np.nanmin(lon_icesat2))
    
    print(np.nanmax(ice_thickness))
    print(np.nanmean(ice_thickness))
    print(np.nanmedian(ice_thickness))
    print(np.nanmin(ice_thickness))
    
    avg_ice_thickness = np.zeros((lon_num,lat_num))
    for i in range(lat_num):
        for j in range(lon_num):
            lat_start_sub = lat_start + lat_step*i
            lat_end_sub = lat_start + lat_step*(i+1)
            lon_start_sub = lon_start + lon_step*j
            lon_end_sub = lon_start + lon_step*(j+1)
            print(lat_start_sub)
            print(lat_end_sub)
            print(lon_start_sub)
            print(lon_end_sub)
            avg_ice_thickness[j][i] = np.nanmean(np.where((lat_icesat2 > lat_start_sub) & (lat_icesat2 < lat_end_sub) & (lon_icesat2 > lon_start_sub) & (lon_icesat2 < lon_end_sub), ice_thickness, float("NaN")))
            print('Avg Thickness is ',avg_ice_thickness[j][i])
            
    print(np.array(avg_ice_thickness).shape)
    
    avg_ice_thickness = np.asarray(avg_ice_thickness)
    
    save_avg = str('Results\\latlon_noiceberg_avg_thick_icesat2_' + year + month + '.npz')
    #np.savez_compressed(save_avg, avg_ice_thickness_svd=avg_ice_thickness)
    np.savez_compressed(save_avg, avg_ice_thickness_svd=avg_ice_thickness)
    '''
    #Load averaged ice thickness data
    #load_avg_name = str('D:\ATL10_2018to2021\\Results\\latlon_noiceberg_xy_mesh_avg_thick_icesat2_' + year + month + '.npz')
    
    load_avg_name = str('Results/latlon_noiceberg_avg_thick_icesat2_' + year + month + '.npz')
    dict_ice_thick_avg = np.load(load_avg_name)
    #x_mesh = dict_ice_thick_avg['x_mesh_svd']
    #y_mesh = dict_ice_thick_avg['y_mesh_svd']
    avg_ice_thickness = dict_ice_thick_avg['avg_ice_thickness_svd']
    try:
    #Load averaged ice concentration data to draw contour lines at 15% and 120% (missing values)
        further_avg_ice_con = np.loadtxt('Data/latlon_further_avg_ice_con' + year + month + '.csv', delimiter=',')
        further_avg_ice_con = np.asarray(further_avg_ice_con)
        
        for i in range(len(further_avg_ice_con)):
            for j in range(len(further_avg_ice_con[i])):
                if pd.isna(further_avg_ice_con[i][j]):
                    further_avg_ice_con[i][j] = 120
    except:
        print('No ice concentration data in ' + month + '/' + year)
    #print("Largest averaged ice thickness is " + str(np.nanmax(avg_ice_thickness)) + " m")
    
    #Start plotting    
    plt.figure()
    plt.pcolormesh(lat_mesh,lon_mesh,avg_ice_thickness,vmin=0,vmax=6)
    plt.xlabel('latitude (°N)')
    plt.ylabel('longitude (°E)')
    plt.title('Amundsen/Bellingshausen Sea Ice Thickness [m] in ' + month + '/' + year)
    plt.colorbar()
    try:
        plt.contour(lat_mesh,lon_mesh,further_avg_ice_con,levels=[15,119],colors=['m','r'])     
    except:
        print('No ice concentration data to plot in ' + month + '/' + year)
    plt.savefig(str('Results/latlon_noiceberg_same_scale_noInterp_IceThickness_LinearApprox' + year + month + '.pdf'), bbox_inches='tight')
    plt.savefig(str('Results/latlon_noiceberg_same_scale_noInterp_IceThickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')
    
    #time3 = datetime.datetime.now()
    #interval23 = time3 - time2
    #print(interval23)

    #Plot on map
    fig=plt.figure(figsize=(12,8))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.gridlines() 
    ax.coastlines()
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    icemap=ax.pcolormesh(lon_mesh,lat_mesh, avg_ice_thickness, vmin = 0, vmax = 6, transform=ccrs.PlateCarree())
    #icemap=ax.pcolormesh(lon_mesh,lat_mesh, avg_ice_thickness, cmap=plt.cm.Greys, transform=ccrs.PlateCarree())
    fig.colorbar(icemap)
    plt.title('Map of Amundsen/Bellingshausen Sea Ice Thickness [m] in ' + month + '/' + year)
    plt.savefig(str('Results/map_latlon_noiceberg_same_scale_noInterp_IceThickness_LinearApprox' + year + month + '.pdf'), bbox_inches='tight')
    plt.savefig(str('Results/map_latlon_noiceberg_same_scale_noInterp_IceThickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')

#thickness_linear_monthly(str('2018'),str('10'))
'''thickness_linear_monthly(str('2018'),str('11'))
thickness_linear_monthly(str('2018'),str('12'))
thickness_linear_monthly(str('2019'),str('01'))
thickness_linear_monthly(str('2019'),str('02'))
thickness_linear_monthly(str('2019'),str('03'))
thickness_linear_monthly(str('2019'),str('04'))
thickness_linear_monthly(str('2019'),str('05'))
thickness_linear_monthly(str('2019'),str('06'))
thickness_linear_monthly(str('2019'),str('07'))
thickness_linear_monthly(str('2019'),str('08'))
thickness_linear_monthly(str('2019'),str('09'))'''
thickness_linear_monthly(str('2019'),str('10'))
'''thickness_linear_monthly(str('2019'),str('11'))
thickness_linear_monthly(str('2019'),str('12'))
thickness_linear_monthly(str('2020'),str('01'))
thickness_linear_monthly(str('2020'),str('02'))
thickness_linear_monthly(str('2020'),str('03'))
thickness_linear_monthly(str('2020'),str('04'))
thickness_linear_monthly(str('2020'),str('05'))
thickness_linear_monthly(str('2020'),str('06'))
thickness_linear_monthly(str('2020'),str('07'))
thickness_linear_monthly(str('2020'),str('08'))
thickness_linear_monthly(str('2020'),str('09'))
thickness_linear_monthly(str('2020'),str('10'))
thickness_linear_monthly(str('2020'),str('11'))
thickness_linear_monthly(str('2020'),str('12'))
thickness_linear_monthly(str('2021'),str('01'))
thickness_linear_monthly(str('2021'),str('02'))
thickness_linear_monthly(str('2021'),str('03'))
thickness_linear_monthly(str('2021'),str('04'))'''