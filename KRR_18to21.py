# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 05:10:13 2021

@author: Acer
"""
#Stage 1: load each feature into a column vector (20x30 = 600 samples or 600 rows)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

kernel_type = 'laplacian'
lat_start = -75
lat_end = -65
lon_start = -142
lon_end = -85

lat_num = 30
lon_num = 20
flatten_dim = lat_num*lon_num

lat_step = (lat_end-lat_start)/lat_num
lon_step = (lon_end-lon_start)/lon_num
lat_array = np.linspace(lat_start,lat_end,lat_num)
lon_array = np.linspace(lon_start,lon_end,lon_num)
lat_mesh, lon_mesh = np.meshgrid(lat_array,lon_array)
#print(lat_mesh[0:5,0:5])
#print(lon_mesh[0:5,0:5])
lat_mesh_flatmonth = np.asarray(lat_mesh.flatten())
lon_mesh_flatmonth = np.asarray(lon_mesh.flatten())
print(lat_mesh_flatmonth.shape)
print(lon_mesh_flatmonth.shape)

lat_mesh_flatfull = []
lon_mesh_flatfull = []
year_flatfull = []
month_flatfull = []
ice_con_flatfull = []
myi_flatfull = []
ice_thick_flatfull = []

# Define a function to get input matrix for each month
def traindata_prep(year,month): 
    year_int = int(year)
    month_int = int(month)
    year_flatmonth = np.full(shape = flatten_dim, fill_value = year_int, dtype = np.int)
    month_flatmonth = np.full(shape = flatten_dim, fill_value = month_int, dtype = np.int)

    #Load ice concentration array
    ice_con_mtrx = np.loadtxt('IceConcentration_2018to2021/Results/latlon_further_avg_ice_con' + year + month + '.csv', delimiter=',')
    ice_con_mtrx = np.asarray(ice_con_mtrx)
    ice_con_flatmonth = ice_con_mtrx.flatten()
    print(ice_con_flatmonth.shape)
    
    #Load myi ratio
    myi_mtrx = np.loadtxt(str('MultiyearIce_2018to2021/Results/myi_onemonth_further_avg_' + year + month + '.csv'), delimiter=',')
    myi_mtrx = np.asarray(ice_con_mtrx)
    myi_flatmonth = myi_mtrx.flatten()
    print(myi_flatmonth.shape)
    
    #Load ice thickness array
    dict_ice_thick_avg = np.load(str('ATL10_2018to2021/Results/latlon_noiceberg_avg_thick_icesat2_' + year + month + '.npz'))
    ice_thick_mtrx = dict_ice_thick_avg['avg_ice_thickness_svd']
    ice_thick_mtrx = np.asarray(ice_thick_mtrx)
    ice_thick_flatmonth = ice_thick_mtrx.flatten()
    print(ice_thick_flatmonth.shape)
    
    #ice_thick_flatfull = np.concatenate((ice_thick_flatfull, ice_thick_flatmonth))
    
    lat_mesh_flatfull.extend(lat_mesh_flatmonth)
    lon_mesh_flatfull.extend(lon_mesh_flatmonth)
    year_flatfull.extend(year_flatmonth)
    month_flatfull.extend(month_flatmonth)
    ice_con_flatfull.extend(ice_con_flatmonth)
    myi_flatfull.extend(myi_flatmonth)
    ice_thick_flatfull.extend(ice_thick_flatmonth)

traindata_prep('2018','10')
traindata_prep('2018','11')

traindata_prep('2019','02')
traindata_prep('2019','03')
traindata_prep('2019','04')
traindata_prep('2019','05')
traindata_prep('2019','06')
traindata_prep('2019','07')
traindata_prep('2019','08')
traindata_prep('2019','09')
traindata_prep('2019','10')
traindata_prep('2019','11')

traindata_prep('2020','02')
traindata_prep('2020','03')
traindata_prep('2020','04')
traindata_prep('2020','05')
traindata_prep('2020','06')
traindata_prep('2020','07')
traindata_prep('2020','08')
traindata_prep('2020','09')
traindata_prep('2020','10')
traindata_prep('2020','11')

#traindata_prep('2021','02')
#traindata_prep('2021','03')
#traindata_prep('2021','04')

lat_mesh_flatfull = np.asarray(lat_mesh_flatfull)
lon_mesh_flatfull = np.asarray(lon_mesh_flatfull)
year_flatfull = np.asarray(year_flatfull)
month_flatfull = np.asarray(month_flatfull)
ice_con_flatfull = np.asarray(ice_con_flatfull)
myi_flatfull = np.asarray(myi_flatfull)
ice_thick_flatfull = np.asarray(ice_thick_flatfull)

print(type(lat_mesh_flatfull))
print(type(year_flatfull))
print(type(myi_flatfull))
print(type(ice_thick_flatfull))
# Stack all columns of relevant data together into a matrix
input_output_mtrx = np.asarray([lat_mesh_flatfull, lon_mesh_flatfull, year_flatfull, month_flatfull, ice_con_flatfull, myi_flatfull, ice_thick_flatfull]).T
#input_output_mtrx = np.asarray([lat_mesh_flatfull, lon_mesh_flatfull, ice_con_flatfull, myi_flatfull, ice_thick_flatfull]).T

print(input_output_mtrx.shape)
print(ice_thick_flatfull.shape)

#print(np.sum(input_output_mtrx)) #check if NaN is still in the matrix
'''
input_output_mtrx_df = pd.DataFrame(data=input_output_mtrx)
input_output_mtrx_df.dropna(inplace=True) #Drop all rows with any NaN value(s) in any column(s)
input_output_mtrx_woNaN = input_output_mtrx_df.to_numpy() #convert back from Pandas data frame to numpy array
print(input_output_mtrx_woNaN.shape)
'''
#Replace all missing (NaN) values in each column with the mean of that column 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(input_output_mtrx)
input_output_mtrx_woNaN = imp_mean.transform(input_output_mtrx)

print(np.sum(input_output_mtrx_woNaN, axis = 0)) #check if NaN is still in any column of the matrix

input_mtrx_woNaN = input_output_mtrx_woNaN[:,0:-1]
ice_thick_woNaN = input_output_mtrx_woNaN[:,-1]
print(input_mtrx_woNaN.shape)
print(ice_thick_woNaN.shape)
mean_thickness = np.mean(ice_thick_woNaN)
#median_thickness = np.median(ice_thick_woNaN)
#print(median_thickness)

#Split traindata into training set and test set
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

#Split data into 60% training set, 20% development set (for hyperparameter-tuning), 20% test set
traindev_data, test_data = train_test_split(input_output_mtrx_woNaN, test_size=0.2)#, random_state=25)
train_data, dev_data = train_test_split(traindev_data, test_size=0.25) #Note: 0.8*0.25 = 0.2
print("No. of training set examples: " + str(train_data.shape[0]))
print("No. of development set examples: " + str(dev_data.shape[0]))
print("No. of testing set examples: " + str(test_data.shape[0]))

train_input = train_data[:,0:-1]
train_output = train_data[:,-1]
dev_input = dev_data[:,0:-1]
dev_output = dev_data[:,-1]
test_input = test_data[:,0:-1]
test_output = test_data[:,-1]
print(train_output.shape)
print(test_input.shape)
#Array of hyperparameters to try out and evaluate validation error
#alpha_try = np.logspace(-10, 0, num=11, base=10)
#gamma_try = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
alpha_try = np.logspace(-10,0, num=11, base = 3)
gamma_try = np.logspace(-10,0, num=11, base = 3)

alpha_mesh, gamma_mesh = np.meshgrid(alpha_try, gamma_try)
print('alpha_mesh.shape ' + str(alpha_mesh.shape))
mesh_shape = alpha_mesh.shape
rms_train_error_array = []
rms_dev_error_array = []
#Convention: alpha on x-axis, gamma on y-axis
for gamma_ele in gamma_try:
    for alpha_ele in alpha_try:    
        KRRmodel = KernelRidge(alpha = alpha_ele, gamma = gamma_ele, kernel = kernel_type)
        KRRmodel.fit(train_input, train_output)
        pred_train_output = KRRmodel.predict(train_input)
        pred_dev_output = KRRmodel.predict(dev_input)
        print(pred_dev_output.shape)
        
        pred_train_output[pred_train_output<0] = 0
        pred_dev_output[pred_dev_output<0] = 0    
        
        rms_train_error = np.sqrt(mean_squared_error(train_output, pred_train_output))
        rms_dev_error = np.sqrt(mean_squared_error(dev_output, pred_dev_output))
        rms_train_error_array += [rms_train_error]
        rms_dev_error_array += [rms_dev_error]
        print('rms_train_error' + str(rms_train_error))
        print('rms_dev_error' + str(rms_dev_error))
        

print(rms_train_error_array)
print(rms_dev_error_array)
rms_train_error_mtrx = np.reshape(rms_train_error_array, (mesh_shape))
rms_dev_error_mtrx = np.reshape(rms_dev_error_array, (mesh_shape))

#rms_trainplustest_error_mtrx = rms_train_error_mtrx + rms_test_error_mtrx
#index_rms_min = np.where(rms_trainplustest_error_mtrx == np.min(rms_trainplustest_error_mtrx))
index_rms_min = np.where(rms_dev_error_mtrx == np.min(rms_dev_error_mtrx))
print(index_rms_min)
alpha_optimized = alpha_mesh[index_rms_min]
gamma_optimized = gamma_mesh[index_rms_min]
print('alpha_optimized' + str(alpha_optimized))
print('gamma_optimized' + str(gamma_optimized))

plt.figure()
plt.pcolormesh(alpha_mesh, gamma_mesh, rms_train_error_mtrx)
plt.colorbar()
plt.title('root mean square training set error of ice thickness[m]')
plt.xlabel('alpha')
plt.ylabel('gamma')

plt.figure()
plt.pcolormesh(alpha_mesh, gamma_mesh, rms_dev_error_mtrx)
plt.colorbar()
plt.title('root mean square development set error of ice thickness[m]')
plt.xlabel('alpha')
plt.ylabel('gamma')

#Apply the model with optimized hyperparameters
KRRmodel_opt = KernelRidge(alpha = alpha_optimized, gamma = gamma_optimized, kernel = kernel_type)
KRRmodel_opt.fit(train_input, train_output)
pred_train_output_opt = KRRmodel_opt.predict(train_input)
pred_dev_output_opt = KRRmodel_opt.predict(dev_input)
pred_test_output_opt = KRRmodel_opt.predict(test_input)
#Since thickness cannot be negative, all negative thickness values returned by KRR are set to 0
pred_train_output_opt[pred_train_output_opt<0] = 0
pred_dev_output_opt[pred_dev_output_opt<0] = 0
pred_test_output_opt[pred_test_output_opt<0] = 0

print(pred_test_output_opt.shape)        
rms_train_error_opt = np.sqrt(mean_squared_error(train_output, pred_train_output_opt))
rms_dev_error_opt = np.sqrt(mean_squared_error(dev_output, pred_dev_output_opt))
rms_test_error_opt = np.sqrt(mean_squared_error(test_output, pred_test_output_opt))
#Compute training, development, and test error in percentage %
relative_train_error_opt = rms_train_error_opt/mean_thickness*100
relative_dev_error_opt = rms_dev_error_opt/mean_thickness*100
relative_test_error_opt = rms_test_error_opt/mean_thickness*100
print('Mean ice thickness = ' + str(mean_thickness) + ' m')
print('rms_train_error_opt = ' + str(rms_train_error_opt) + ' m')
print('rms_dev_error_opt = ' + str(rms_dev_error_opt) + ' m')
print('rms_test_error_opt = ' + str(rms_test_error_opt) + ' m')
print('relative_train_error_opt = ' +str (relative_train_error_opt) + '%')
print('relative_dev_error_opt = ' + str(relative_dev_error_opt) + '%')
print('relative_test_error_opt = ' + str(relative_test_error_opt) + '%')


#Compare the output (ice thickness) in true test data v. predicted test data
def apply_cloudyMonth(year, month): 
    year_flatmonth = np.full(shape = flatten_dim, fill_value = year, dtype = np.int)
    month_flatmonth = np.full(shape = flatten_dim, fill_value = month, dtype = np.int)
    #year_flatmonth = np.reshape(np.full(shape = flatten_dim, fill_value = year, dtype = np.int), (flatten_dim,1))
    #month_flatmonth = np.reshape(np.full(shape = flatten_dim, fill_value = month, dtype = np.int), (flatten_dim,1))
    print(year_flatmonth.shape)
    print(month_flatmonth.shape)
    #Load ice concentration array
    ice_con_mtrx = np.loadtxt('IceConcentration_2018to2021/Results/latlon_further_avg_ice_con' + year + month + '.csv', delimiter=',')
    ice_con_flatmonth = ice_con_mtrx.flatten()
    #ice_con_mtrx = np.reshape((ice_con_mtrx), (flatten_dim,1))
    #ice_con_flatmonth = np.reshape(ice_con_mtrx.flatten(), (flatten_dim,1))
    print(ice_con_flatmonth.shape)
    
    #Load myi ratio
    myi_mtrx = np.loadtxt(str('MultiyearIce_2018to2021/Results/myi_onemonth_further_avg_' + year + month + '.csv'), delimiter=',')
    myi_mtrx = np.asarray(ice_con_mtrx)
    myi_flatmonth = myi_mtrx.flatten()
    #myi_flatmonth = np.reshape(myi_mtrx.flatten(), (flatten_dim,1))
    print(myi_flatmonth.shape)
    
    #Load ice thickness array
    dict_ice_thick_avg = np.load(str('ATL10_2018to2021/Results/latlon_noiceberg_avg_thick_icesat2_' + year + month + '.npz'))
    ice_thick_mtrx = dict_ice_thick_avg['avg_ice_thickness_svd']
    ice_thick_mtrx = np.asarray(ice_thick_mtrx)
    ice_thick_flatmonth = ice_thick_mtrx.flatten()
    #ice_thick_flatmonth = np.reshape(ice_thick_mtrx.flatten(), (flatten_dim,1))
    print(ice_thick_flatmonth.shape)
    
    print(lat_mesh_flatmonth.shape)
    print(lon_mesh_flatmonth.shape)
    print(type(lat_mesh_flatmonth))
    print(type(year_flatmonth))
    print(type(myi_flatmonth))
    
   # year_flatmonth = np.asarray(year_flatmonth)
    #month_flatmonth = np.asarray(month_flatmonth)
    #ice_con_flatfull = np.asarray(ice_con_flatmonth)
    #myi_flatmonth = np.asarray(myi_flatmonth)
    cloudy_month_input = np.asarray([lat_mesh_flatmonth, lon_mesh_flatmonth, year_flatmonth, month_flatmonth, ice_con_flatmonth, myi_flatmonth]).T
    #print(cloudy_month_input)
    print(cloudy_month_input.shape)
    '''
    cloudy_month_input_df = pd.DataFrame(data=cloudy_month_input)
    cloudy_month_input_df.dropna(inplace=True) #Drop all rows with any NaN value(s) in any column(s)
    cloudy_month_input_woNaN = cloudy_month_input_df.to_numpy()
    print(cloudy_month_input_woNaN.shape)
    '''
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(cloudy_month_input)
    cloudy_month_input_woNaN = imp_mean.transform(cloudy_month_input)    
    
    #cloudy_month_input_woNaN = cloudy_month_input
    lat_mesh_1D = cloudy_month_input_woNaN[:,0]
    lon_mesh_1D = cloudy_month_input_woNaN[:,1]
    pred_cloundy_month_output_1D = KRRmodel_opt.predict(cloudy_month_input_woNaN)
    lat_mesh_2D = np.reshape(lat_mesh_1D, (lon_num, lat_num))
    lon_mesh_2D = np.reshape(lon_mesh_1D, (lon_num, lat_num))
    pred_cloundy_month_output_2D = np.reshape(pred_cloundy_month_output_1D, (lon_num, lat_num))
    print(lat_mesh_1D.shape)
    print(lon_mesh_1D.shape)
    print(pred_cloundy_month_output_1D.shape)

    #Load averaged ice concentration data to draw contour lines at 15% and 120% (missing values)
    further_avg_ice_con = np.loadtxt('ATL10_2018to2021/Data/latlon_further_avg_ice_con' + year + month + '.csv', delimiter=',')
    further_avg_ice_con = np.asarray(further_avg_ice_con)
    for i in range(len(further_avg_ice_con)):
        for j in range(len(further_avg_ice_con[i])):
            if pd.isna(further_avg_ice_con[i][j]):
                further_avg_ice_con[i][j] = 120
    #Filter out cells with too small ice concentration (below 15%) or land (marked as 120%)
    pred_cloundy_month_output_2D_filtered = np.where((15 < further_avg_ice_con) & (further_avg_ice_con < 120), pred_cloundy_month_output_2D, float('NaN'))
    #Start plotting
    plt.figure()
    plt.pcolormesh(lat_mesh_2D, lon_mesh_2D, pred_cloundy_month_output_2D_filtered, vmin=0, vmax=6)
    plt.colorbar()
    plt.contour(lat_mesh,lon_mesh,further_avg_ice_con,levels=[15,119],colors=['m','r'])     
    plt.title('Ice thickness [m] predicted by KRR model in ' + month + '/' + year)
    plt.xlabel('latitude (°N)')
    plt.ylabel('longitude (°E)')
    plt.savefig(str('latlon_noiceberg_same_scale_KRR_IceThickness_LinearApprox' + year + month + '.pdf'))
    plt.savefig(str('latlon_noiceberg_same_scale_KRR_IceThickness_LinearApprox' + year + month + '.png'))
    
apply_cloudyMonth('2018', '10')
apply_cloudyMonth('2019', '07')

#linear kernel -> rms error around 57-63%
#polynomial kernel -> rms error around 
#laplacian kernel -> relative rms error 0.32 m aka 26.8% (train error), 0.48 m aka 40.3% (dev error), 0.53 m aka 44.1% (test error)
#sigmoid kernel -> rms error around 0.79 m aka 60-70%