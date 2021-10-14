# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:52:23 2021

@author: amill
"""

#Kalman filter example - turning car. Example 9 taken from https://www.kalmanfilter.net/multiExamples.html

#import dependencies 

import numpy as np 
import matplotlib.pyplot as plt 

#setting the constants 
delta_t = 1
sigma_a_sqrd = 0.15
sigma_x = 3
sigma_y = 3
F = np.array([[1, delta_t, 0.5*(delta_t**2), 0, 0, 0], 
     [0, 1, delta_t, 0, 0, 0], 
     [0, 0, 1, 0, 0, 0], 
     [0, 0, 0, 1, delta_t, 0.5*(delta_t**2)], 
     [0, 0, 0, 0, 1, delta_t],
     [0, 0, 0, 0, 0, 1]])
Q = np.array([[(delta_t**4)/4, (delta_t**3)/2, (delta_t**2)/2, 0, 0, 0], 
     [(delta_t**3)/2, delta_t**2, delta_t, 0, 0, 0], 
     [(delta_t**2)/2, delta_t, 1, 0, 0, 0], 
     [0, 0, 0, (delta_t**4)/4, (delta_t**3)/2, (delta_t**2)/2], 
     [0, 0, 0, (delta_t**3)/2, delta_t**2, delta_t],
     [0, 0, 0, (delta_t**2)/2, delta_t, 1]])*sigma_a_sqrd
R_n = np.array([[sigma_x**2, 0], 
                [0, sigma_y**2]])
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]])

measure_x = np.array([[-393.66], [-375.93], [-351.04], [-328.96], [-299.35], [-273.36], [-245.89], [-222.58], [-198.03], [-174.17], [-146.32], [-123.72], [-103.47], [-78.23], [-52.63], [-23.34], [25.96], [49.72], [76.94], [95.38], [119.83], [144.01], [161.84], [180.56], [201.42], [222.62], [239.4], [252.51], [266.26], [271.75], [277.4], [294.12], [301.23], [291.8], [299.89]])
measure_y = np.array([[300.4], [301.78], [295.1], [305.19], [301.06], [302.05], [300], [303.57], [296.33], [297.65], [297.41], [299.61], [299.6], [302.39], [295.04], [300.09], [294.72], [298.61], [294.64], [284.88], [272.82], [264.93], [251.46], [241.27], [222.98], [203.73], [184.1], [166.12], [138.71], [119.71], [100.41], [79.76], [50.62], [32.99], [2.14]])
measurements = np.hstack((measure_x, measure_y) )

#initialization / iteration 0 
#don't know the vehcile location so we set it = 0 for now. 

initial_pos = np.array([[0], [0], [0], [0], [0], [0]])

estimate_uncertainty = np.array([[500, 0, 0, 0, 0, 0], 
                                [0, 500, 0, 0, 0, 0],
                                [0, 0, 500, 0, 0, 0],
                                [0, 0, 0, 500, 0, 0],
                                [0, 0, 0, 0, 500, 0],
                                [0, 0, 0, 0, 0, 500]])

new_predicted_position = np.dot(F, initial_pos)

new_predicted_position_uncertainty = np.dot(np.dot(F, estimate_uncertainty), (np.transpose(F))) + Q

#print(measurements[0])

#pulling out each measurement 
def select_measure(measurements, i):
    new_measurement = measurements[i]
    return new_measurement 

#equations needed for the kalman filter 

def new_predicted_position_func(new_predicted_position, F):
    new_predicted_position = np.dot(F, new_predicted_position)
    return new_predicted_position


def new_predicted_position_uncertainty_func(F, new_predicted_position_uncertainty, Q):
    new_predicted_position_uncertainty = np.dot(F, np.dot(new_predicted_position_uncertainty, (np.transpose(F)))) + Q
    return new_predicted_position_uncertainty

def Kalman_gain_func(new_predicted_position_uncertainty, H, R_n):
    Kalman_gain = np.dot(np.dot(new_predicted_position_uncertainty, np.transpose(H)), np.linalg.inv(np.dot(np.dot(H, new_predicted_position_uncertainty),np.transpose(H)) + R_n))
    return Kalman_gain

def new_current_state_func(new_predicted_position, Kalman_gain, H, new_measurement):
    new_current_state = new_predicted_position + np.dot(Kalman_gain,(new_measurement - np.dot(H, new_predicted_position)))
    return new_current_state

def current_estimate_uncertainty_func(H, R_n, Kalman_gain, new_predicted_position_uncertainty):
    current_estimate_uncertainty = np.dot((np.identity(6) - np.dot(Kalman_gain, H)), np.dot(new_predicted_position_uncertainty, np.transpose((np.identity(6) - np.dot(Kalman_gain, H))))) + np.dot(Kalman_gain, np.dot(R_n, np.transpose(Kalman_gain)))
    return current_estimate_uncertainty

#now the for loop to do iterations 1-35. 
current_state_xpos_estimate = []
current_state_ypos_estimate = []

for i in range(len(measurements)):
    new_measurement = select_measure(measurements, i)
    Kalman_gain = Kalman_gain_func(new_predicted_position_uncertainty, H, R_n)
    new_current_state = new_current_state_func(new_predicted_position, Kalman_gain, H, new_measurement)
    #annoying I have to do the next part, need to ask why 
    new_current_state = np.array([[new_current_state[0][0]], [new_current_state[1][0]], [new_current_state[2][0]], [new_current_state[3][1]], [new_current_state[4][1]], [new_current_state[5][1]]])
    current_state_xpos_estimate.append(new_current_state[0])
    current_state_ypos_estimate.append(new_current_state[3])
    current_estimate_uncertainty = current_estimate_uncertainty_func(H, R_n, Kalman_gain, new_predicted_position_uncertainty)
    new_predicted_position = new_predicted_position_func(new_current_state, F)
    new_predicted_position_uncertainty = new_predicted_position_uncertainty_func(F, current_estimate_uncertainty, Q)

#some of my uncertainties are different to the example but unsure if this is because of floating point while the example rounds. 
plt.scatter(measure_x, measure_y)
plt.plot(measure_x, measure_y)
plt.scatter(current_state_xpos_estimate, current_state_ypos_estimate)
plt.plot(current_state_xpos_estimate, current_state_ypos_estimate)
plt.title('Measured (blue) and Estimated (yellow) Route')
plt.xlabel('X(m)')
plt.ylabel('Y(m)')
plt.show()
#matplotlib charts 