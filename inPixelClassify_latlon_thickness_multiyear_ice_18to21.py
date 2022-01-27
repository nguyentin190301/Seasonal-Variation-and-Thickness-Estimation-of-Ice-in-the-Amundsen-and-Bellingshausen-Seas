# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:10:35 2021

@author: Acer
"""
import numpy as np
import matplotlib.pyplot as plt

lat_start = -75
lat_end = -65
lon_start = -142
lon_end = -85

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
print(lat_mesh.shape)
print(lon_mesh.shape)

period_thickness_myi_array = []
period_thickness_fyi_array = []
month_year_array = []

def inPixelClassify(year,month):
    '''
    myi_fn = str('MultiyearIce_2018to2021/Results/myi_onemonth_further_avg_' + year + month + '.csv')
    avg_myi = np.loadtxt(myi_fn, delimiter=',')
    avg_myi = np.array(avg_myi)
    avg_myi = np.where(avg_myi==0, float("NaN"), avg_myi) #drop all myi = 0 values
    
    save_whole = str('ATL10_2018to2021/Results/xyandlatlonGrid_thick_icesat2_' + year + month + '.npz')
    dict_xyGrid_thick_icesat2 = np.load(save_whole)
    #dict_xyGrid_thick_icesat2 = np.load('D:\ATL10_2018to2021\\' + save_whole)
    lat_icesat2 = dict_xyGrid_thick_icesat2['latitude_icesat2_svd']
    lon_icesat2 = dict_xyGrid_thick_icesat2['longitude_icesat2_svd']
    ice_thickness = dict_xyGrid_thick_icesat2['ice_thickness_svd']
    
    ice_thickness = np.array(ice_thickness)
    ice_thickness_dim = ice_thickness.shape[0]
    for i in range(ice_thickness_dim):
        if ice_thickness[i] > iceberg_thickness_cap: #avoid iceberg
            ice_thickness[i] = float("NaN")  

    pixel_thickness_avg_mtrx_myi = np.zeros((lon_num, lat_num))
    pixel_thickness_avg_mtrx_fyi = np.zeros((lon_num, lat_num))
    
    max_mtrx = np.zeros((lon_num, lat_num))
    
    monthly_thickness_myi_array = []
    monthly_thickness_fyi_array = []
    
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
            pixel_thickness_array_withNaN = np.where((lat_icesat2 > lat_start_sub) & (lat_icesat2 < lat_end_sub) & (lon_icesat2 > lon_start_sub) & (lon_icesat2 < lon_end_sub), ice_thickness, float("NaN"))
            pixel_thickness_array_noNaN = pixel_thickness_array_withNaN[~np.isnan(pixel_thickness_array_withNaN)]
            print('pixel_thickness_array_noNaN.shape' + str(pixel_thickness_array_noNaN.shape))
            try: #Use a try and except loop to avoid error when encountering a no-element array
                print('min = ' + str(np.min(pixel_thickness_array_noNaN)))
                print('max = ' + str(np.max(pixel_thickness_array_noNaN)))
                
                max_mtrx[j][i] = np.max(pixel_thickness_array_noNaN)
                
                #Reorder thickness array to get thicker (myi) first, thinner (fyi) last
                pixel_thickness_array_noNaN_decreasing = np.asarray(sorted(pixel_thickness_array_noNaN, reverse = True))
                pixel_thickness_array_noNaN_length = pixel_thickness_array_noNaN.shape[0]
                print('pixel_thickness_array_noNaN_length' + str(pixel_thickness_array_noNaN_length))
                #get the index separating myi (thicker, earlier array members) and fyi (thinner, later array members)
                pixel_thickness_array_noNaN_length_myi = int(pixel_thickness_array_noNaN_length*avg_myi[j][i]/100)
                print('pixel_thickness_array_noNaN_length_myi' + str(pixel_thickness_array_noNaN_length_myi))
                pixel_thickness_array_noNaN_myi = np.asarray(pixel_thickness_array_noNaN_decreasing[:pixel_thickness_array_noNaN_length_myi])
                pixel_thickness_array_noNaN_fyi = np.asarray(pixel_thickness_array_noNaN_decreasing[pixel_thickness_array_noNaN_length_myi:])
                print('first ele = ' + str(pixel_thickness_array_noNaN_decreasing[0]))
                print('last ele = ' + str(pixel_thickness_array_noNaN_decreasing[-1]))
                print('first ele of thick = ' + str(pixel_thickness_array_noNaN_myi[0]))
                print('last ele of thick = ' + str(pixel_thickness_array_noNaN_myi[-1]))
                print('first ele of thin = ' + str(pixel_thickness_array_noNaN_fyi[0]))
                print('last ele of thin = ' + str(pixel_thickness_array_noNaN_fyi[-1]))
                #print('total shape check = ' + str(pixel_thickness_array_noNaN_myi.shape + pixel_thickness_array_noNaN_fyi.shape))
                pixel_thickness_avg_myi = np.nanmean(pixel_thickness_array_noNaN_myi)
                pixel_thickness_avg_fyi = np.nanmean(pixel_thickness_array_noNaN_fyi)
                print('pixel_thickness_avg_myi' + str(pixel_thickness_avg_myi))
                print('pixel_thickness_avg_fyi' + str(pixel_thickness_avg_fyi))
                pixel_thickness_avg_mtrx_myi[j][i] = pixel_thickness_avg_myi
                pixel_thickness_avg_mtrx_fyi[j][i] = pixel_thickness_avg_fyi
                #monthly_thickness_myi_array += pixel_thickness_array_noNaN_myi
                #monthly_thickness_fyi_array += pixel_thickness_array_noNaN_fyi 
                print('ya1')
                #monthly_thickness_myi_array.append(pixel_thickness_array_noNaN_myi)
                #monthly_thickness_fyi_array.append(pixel_thickness_array_noNaN_fyi)
                #monthly_thickness_myi_array = np.concaternate([monthly_thickness_myi_array, pixel_thickness_array_noNaN_myi])
                #monthly_thickness_fyi_array = np.concaternate([monthly_thickness_fyi_array, pixel_thickness_array_noNaN_fyi])
                monthly_thickness_myi_array.extend(pixel_thickness_array_noNaN_myi)
                monthly_thickness_fyi_array.extend(pixel_thickness_array_noNaN_fyi)
                print('ya2')
                myi_shape = np.asarray(monthly_thickness_myi_array).shape
                fyi_shape = np.asarray(monthly_thickness_fyi_array).shape
                print('ya3')
                print('monthly_thickness_myi_array.shape = ' + str(myi_shape))
                print('monthly_thickness_fyi_array.shape = ' + str(fyi_shape))
                print('ya4')
            except:
                continue
            
    np.savetxt(str('CrossDataset_Results/pixel_thickness_avg_myi_' + year + month + '.csv'), pixel_thickness_avg_mtrx_myi, delimiter=',')
    np.savetxt(str('CrossDataset_Results/pixel_thickness_avg_fyi_' + year + month + '.csv'), pixel_thickness_avg_mtrx_fyi, delimiter=',')
    np.savetxt(str('CrossDataset_Results/max_mtrx' + year + month + '.csv'), max_mtrx, delimiter=',')
    '''
    pixel_thickness_avg_mtrx_myi = np.loadtxt(str('CrossDataset_Results/pixel_thickness_avg_myi_' + year + month + '.csv'), delimiter=',')
    pixel_thickness_avg_mtrx_fyi = np.loadtxt(str('CrossDataset_Results/pixel_thickness_avg_fyi_' + year + month + '.csv'), delimiter=',')
    
    #Convert all elements which equal 0 to NaN
    pixel_thickness_avg_mtrx_myi = np.where(pixel_thickness_avg_mtrx_myi == 0, np.nan, pixel_thickness_avg_mtrx_myi)
    pixel_thickness_avg_mtrx_fyi = np.where(pixel_thickness_avg_mtrx_fyi == 0, np.nan, pixel_thickness_avg_mtrx_fyi)
    
    plt.figure()
    plt.pcolormesh(lat_mesh,lon_mesh, pixel_thickness_avg_mtrx_myi,vmin=0,vmax=10)
    plt.xlabel('latitude (°N)')
    plt.ylabel('longitude (°E)')
    plt.title('Amundsen/Bellingshausen Sea MYI Thickness [m] in ' + month + '/' + year)
    plt.colorbar()
    '''try:
        plt.contour(lat_mesh,lon_mesh,further_avg_ice_con,levels=[15,119],colors=['m','r'])     
    except:
        print('No ice concentration data to plot in ' + month + '/' + year)'''
    plt.savefig(str('CrossDataset_Results/latlon_noiceberg_MYI_Thickness_LinearApprox' + year + month + '.pdf'), bbox_inches='tight')
    plt.savefig(str('CrossDataset_Results/latlon_noiceberg_MYI_Thickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')
    plt.savefig(str('CrossDataset_Results/MYI_Thickness_Plot/latlon_noiceberg_MYI_Thickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')        
    
    plt.figure()
    plt.pcolormesh(lat_mesh,lon_mesh, pixel_thickness_avg_mtrx_fyi)
    plt.xlabel('latitude (°N)')
    plt.ylabel('longitude (°E)')
    plt.title('Amundsen/Bellingshausen Sea FYI Thickness [m] in ' + month + '/' + year)
    plt.colorbar()
    '''try:
        plt.contour(lat_mesh,lon_mesh,further_avg_ice_con,levels=[15,119],colors=['m','r'])     
    except:
        print('No ice concentration data to plot in ' + month + '/' + year)'''
    plt.savefig(str('CrossDataset_Results/latlon_noiceberg_FYI_Thickness_LinearApprox' + year + month + '.pdf'), bbox_inches='tight')
    plt.savefig(str('CrossDataset_Results/latlon_noiceberg_FYI_Thickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')
    plt.savefig(str('CrossDataset_Results/FYI_Thickness_Plot/latlon_noiceberg_FYI_Thickness_LinearApprox' + year + month + '.png'), bbox_inches='tight')    
    '''
    monthly_thickness_myi_avg = np.nanmean(monthly_thickness_myi_array)
    monthly_thickness_fyi_avg = np.nanmean(monthly_thickness_fyi_array)
    print('monthly_thickness_myi_avg = ' + str(monthly_thickness_myi_avg))
    print('monthly_thickness_fyi_avg = ' + str(monthly_thickness_fyi_avg))
    
    period_thickness_myi_array.append(monthly_thickness_myi_avg)
    period_thickness_fyi_array.append(monthly_thickness_fyi_avg)
    '''
    month_year = str(month + '/' + year)
    month_year_array.append(month_year)
    
#inPixelClassify('2019','08')
inPixelClassify('2018','10')
inPixelClassify('2018','11')

inPixelClassify('2019','02')
inPixelClassify('2019','03')
inPixelClassify('2019','04')

inPixelClassify('2019','05')
inPixelClassify('2019','06')
inPixelClassify('2019','07')
inPixelClassify('2019','08')
inPixelClassify('2019','09')
inPixelClassify('2019','10')
inPixelClassify('2019','11')

inPixelClassify('2020','02')
inPixelClassify('2020','03')
inPixelClassify('2020','04')
inPixelClassify('2020','05')
inPixelClassify('2020','06')
inPixelClassify('2020','07')
inPixelClassify('2020','08')
inPixelClassify('2020','09')
inPixelClassify('2020','10')
inPixelClassify('2020','11')

inPixelClassify('2021','02')
inPixelClassify('2021','03')
inPixelClassify('2021','04')

np.savetxt(str('CrossDataset_Results/period_thickness_myi_array' + '.csv'), period_thickness_myi_array, delimiter=',')
np.savetxt(str('CrossDataset_Results/period_thickness_fyi_array' + '.csv'), period_thickness_fyi_array, delimiter=',')

'''
plt.figure()
plt.plot(month_year_array, period_thickness_myi_array)
plt.xticks(rotation=90)
plt.title('MYI thickness [m] averaged over the region')    
plt.xlabel('Month')
plt.ylabel('Multi-year ice thickness [m]')
plt.savefig('CrossDataset_Results/inPixelClassify_MYI_thickness_gridavg_18to21.png', bbox_inches='tight')
plt.savefig('CrossDataset_Results/inPixelClassify_MYI_thickness_gridavg_18to21.pdf', bbox_inches='tight')

plt.figure()
plt.plot(month_year_array, period_thickness_fyi_array)
plt.xticks(rotation=90)
plt.title('FYI thickness [m] averaged over the region')    
plt.xlabel('Month')
plt.ylabel('First-year ice thickness [m]')
plt.savefig('CrossDataset_Results/inPixelClassify_FYI_thickness_gridavg_18to21.png', bbox_inches='tight')
plt.savefig('CrossDataset_Results/inPixelClassify_FYI_thickness_gridavg_18to21.pdf', bbox_inches='tight')
'''