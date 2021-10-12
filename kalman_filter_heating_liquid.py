# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:43:01 2021

@author: amill
"""

#Kalman Filter 1D application from https://www.kalmanfilter.net/kalman1d.html
#ESTIMATING THE TEMPERATURE OF A HEATING LIQUID

#importing dependencies 
import matplotlib.pyplot as plt 

#measured and true values 
true_temp = [50.479, 51.025, 51.5, 52.003, 52.494, 53.002, 53.499, 54.006, 54.498, 54.991]
measured_temp = [50.45, 50.967, 51.6, 52.106, 52.492, 52.819, 53.433, 54.007, 54.523, 54.99]

#this is the initialization step 
initial_guess = 10 
#q is process noise, if we lower this number, we get a better estimate but we basically just give 
#more weighting to the measurement. Which defeats the point of heaving a Kalman filter. 
#So we really need a better model to describe it.  
q = 0.0001
measurement_error = 0.1
interval = 5
estimate_error = 100

estimate_uncertainty = estimate_error**2

#prediction step 
#this model has constant dynamics so predicted estimate = current estimate 
current_estimate = 10 

#extrapolated estimate uncertainty 
eeu = estimate_uncertainty + q

#defining the 5 equations for the next estimate. These are the 5 equations that make up the Kalman Filter 

def Kalman_gain(eeu, measurement_error):
    K = eeu/(eeu + measurement_error**2)
    return K 

def updated_guess(current_estimate, K, measured_temp_index):
    current_estimate = current_estimate + K*(measured_temp_index - current_estimate)
    return current_estimate

def update_estimate_uncertainty(K, eeu):
    estimate_uncertainty = (1-K)*eeu
    return estimate_uncertainty

def predicted_estimate(current_estimate):
    return current_estimate 

def new_eeu(estimate_uncertainty, q):
    eeu = estimate_uncertainty + q
    return eeu 

#Now we do iterations 1 through 10 
predicted_temp = []
Kalman_gain_list = []
for i in range(len(measured_temp)):
    measured_temp_index = measured_temp[i]
    K = Kalman_gain(eeu, measurement_error)
    Kalman_gain_list.append(K)
    current_estimate = updated_guess(current_estimate, K, measured_temp_index)
    predicted_temp.append(current_estimate)
    estimate_uncertainty = update_estimate_uncertainty(K, eeu) 
    current_estimate = predicted_estimate(current_estimate)
    eeu = new_eeu(estimate_uncertainty, q)

measurement_num = []
for i in range(len(measured_temp)):
    measurement_num.append(i)

plt.scatter(measurement_num, true_temp)
plt.scatter(measurement_num, measured_temp)
plt.scatter(measurement_num, predicted_temp)
plt.plot(measurement_num, true_temp)
plt.plot(measurement_num, measured_temp)
plt.plot(measurement_num, predicted_temp)
plt.title('Liquid Temperature')
plt.xlabel('Measurement Number')
plt.ylabel('Temperature (C)')
plt.show()

plt.scatter(measurement_num, Kalman_gain_list)
plt.plot(measurement_num, Kalman_gain_list)
plt.title('Kalman Gain')
plt.xlabel('Measurement Number')
plt.ylabel('Kalman Gain')
plt.show()