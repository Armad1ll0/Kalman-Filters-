# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:46:09 2021

@author: amill
"""

#EXAMPLE 10 – ROCKET ALTITUDE ESTIMATION using Kalman filter taken from https://www.kalmanfilter.net/multiExamples.html

#import dependencies 

import numpy as np 
import matplotlib.pyplot as plt 

#setting the constants 
delta_t = 0.25
acceleration = 30
sigma_x = 20
epsilon = 0.1
g = 9.8

#transition matrix 
F = np.array([[1, delta_t], 
              [0, 1]])

#control matrix 
G = np.array([[0.5*delta_t**2], 
              [delta_t]])

#process noise matrix 
Q = np.array([[(delta_t**4)/4, (delta_t**3)/2], 
              [(delta_t**3)/2, delta_t**2]])*(epsilon**2)

#measurement uncertainty 
R = sigma_x**2

H = np.array([1, 0])

altitude_measurement = np.array([[-32.4], [-11.1], [18],	[22.9],	[19.5],	[28.5],	[46.5],	[68.9],	[48.2],	[56.1],	[90.5],	[104.9], [140.9], [148], [187.6], [209.2], [244.6],	[276.4], [323.5], [357.3], [357.4],	[398.3], [446.7], [465.1], [529.4],	[570.4], [636.8], [693.3], [707.3], [748.5]])
acceleration_measurement = np.array([[39.72], [40.02], [39.97], [39.81], [39.75], [39.6], [39.77], [39.83], [39.73], [39.87], [39.81], [39.92], [39.78], [39.98], [39.76], [39.86], [39.61], [39.86], [39.74], [39.87], [39.63], [39.67], [39.96], [39.8], [39.89], [39.85], [39.9], [39.81], [39.81], [39.68]])

#0th iteration 
#initialisation 
initial_pos = np.array([[0], [0]])

u_0 = g

estimate_uncertainty = np.array([[500, 0], 
              [0, 500]])

#0th prediciton 
new_predicted_position = np.dot(F, initial_pos) + np.dot(G, u_0)

new_predicted_position_uncertainty = np.dot(np.dot(F, estimate_uncertainty), (np.transpose(F))) + Q


#pulling out each measurement 
def select_altitude(altitude_measurement, i):
    new_measurement = altitude_measurement[i]
    return new_measurement 

def select_acceleration(acceleration_measurement, i):
    new_acceleration = acceleration_measurement[i]
    return new_acceleration 

#equations needed for the kalman filter 

def new_predicted_position_func(new_predicted_position, F, new_acceleration, G):
    new_predicted_position = np.dot(F, new_predicted_position) + G*new_acceleration
    return new_predicted_position


def new_predicted_position_uncertainty_func(F, new_predicted_position_uncertainty, Q):
    new_predicted_position_uncertainty = np.dot(F, np.dot(new_predicted_position_uncertainty, (np.transpose(F)))) + Q
    return new_predicted_position_uncertainty

def Kalman_gain_func(new_predicted_position_uncertainty, H, R_n):
    Kalman_gain = np.dot(np.dot(new_predicted_position_uncertainty, np.transpose(H)), (np.dot(np.dot(H, new_predicted_position_uncertainty),np.transpose(H)) + R)**(-1))
    return Kalman_gain

def new_current_state_func(new_predicted_position, Kalman_gain, H, new_measurement):
    A = new_measurement - np.dot(H, new_predicted_position)
    B = Kalman_gain*A
    new_current_state = new_predicted_position + B
    return new_current_state

def current_estimate_uncertainty_func(H, R_n, Kalman_gain, new_predicted_position_uncertainty):
    current_estimate_uncertainty = np.dot((np.identity(2) - np.dot(Kalman_gain, H)), np.dot(new_predicted_position_uncertainty, np.transpose((np.identity(2) - np.dot(Kalman_gain, H))))) + np.dot(Kalman_gain, np.dot(R_n, np.transpose(Kalman_gain)))
    return current_estimate_uncertainty

#now the for loop to do iterations 1-35. 
Kalman_gain_list = []
current_state_list = []
time_list = []

for i in range(len(altitude_measurement)):
    new_measurement = select_altitude(altitude_measurement, i)
    new_acceleration = select_acceleration(acceleration_measurement, i) 
    Kalman_gain = Kalman_gain_func(new_predicted_position_uncertainty, H, R)
    Kalman_gain_list.append(Kalman_gain[0])
    time_list.append(delta_t*i)
    new_current_state = new_current_state_func(new_predicted_position, Kalman_gain, H, new_measurement)
    current_state_list.append(new_current_state[0][0])
    new_current_state = np.array([[new_current_state[0][0]], [new_current_state[1][1]]])
    current_estimate_uncertainty = current_estimate_uncertainty_func(H, R, Kalman_gain, new_predicted_position_uncertainty)
    new_predicted_position = new_predicted_position_func(new_current_state, F, new_acceleration, G)
    new_predicted_position_uncertainty = new_predicted_position_uncertainty_func(F, current_estimate_uncertainty, Q)

#graphs for visualization 

plt.plot(time_list, Kalman_gain_list)
plt.title('Kalman Gain Change Over Time ')
plt.xlabel('Time (s)')
plt.ylabel('Kalman Gain')
plt.show()

measured_altitude_list = []
for i in altitude_measurement:
    measured_altitude_list.append(i)

#note, this is not entirely correct 
plt.plot(time_list, current_state_list)
plt.scatter(time_list, current_state_list)
plt.plot(time_list, measured_altitude_list)
plt.scatter(time_list, measured_altitude_list)
plt.title('Rocket Altitude, Measured (Blue), Estimated (Yellow)')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.show()

