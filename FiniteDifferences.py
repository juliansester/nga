# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:10:59 2021

@author: Julian Sester
"""

import numpy as np
from tqdm import tqdm 

def linear_pde(a_0,a_1,b_0,b_1,gamma,payoff,
               start_value_from =0,
               start_value_to =1,
               Nr_grid_s = 10,
               Nr_grid_t = 100, 
               time_from =0, 
               time_to = 1):
    # Initialize grids
    s = np.linspace(start_value_from,start_value_to,Nr_grid_s)
    Delta_s = s[1]-s[0]
    t = np.linspace(time_from,time_to,Nr_grid_t)
    Delta_t = t[1]-t[0]
    if Delta_s**2/max([a_0+a_1*val*(val>0) for val in s]) < Delta_t:
        print("WARNING: Stability condition is violated. \nChoose more time iterations!")
    #Delta_t = t[1]-t[0]
    u = np.zeros((len(t),len(s)))
    
    #Boundary Condition for maturity T
    u[-1,:] = [payoff(val) for val in s]
    #Boundary Condition for border of space:
    u[:,0] = [payoff(s[0])]*len(t)
    u[:,-1] = [payoff(s[-1])]*len(t)
    time_step = 0
    for i in tqdm(range(len(t)-1,0,-1)):
        time_step += 1
        u[i-1,0] = u[i,0]+(b_0+b_1*s[0])*(Delta_t/(Delta_s))*(u[i,1]-u[i,0])\
                  +0.5*((a_0+a_1*(s[0]*(s[0]>0)))**(2*gamma))*(Delta_t/Delta_s**2)*(u[i,2]-2*u[i,1]+u[i,0])
        u[i-1,-1] = u[i,-1]+(b_0+b_1*s[-1])*(Delta_t/(Delta_s))*(u[i,-1]-u[i,-2])\
              +0.5*((a_0+a_1*(s[-1]*(s[-1]>0)))**(2*gamma))*(Delta_t/Delta_s**2)*(u[i,-1]-2*u[i,-2]+u[i,-3])
        for j in range(1,len(s)-1):
             u[i-1,j] = u[i,j]+(b_0+b_1*s[j])*(Delta_t/(2*Delta_s))*(u[i,j+1]-u[i,j-1])\
                  +0.5*((a_0+a_1*(s[j]*(s[j]>0)))**(2*gamma))*(Delta_t/Delta_s**2)*(u[i,j+1]-2*u[i,j]+u[i,j-1])
    return u[0,:], u

def nonlinear_pde(a_0_lower_bound,a_0_upper_bound,a_1_lower_bound,a_1_upper_bound,
                  b_0_lower_bound,b_0_upper_bound,b_1_lower_bound,b_1_upper_bound,
                  gamma_lower_bound,gamma_upper_bound,
                  payoff,
               start_value_from =0,
               start_value_to =1,
               Nr_grid_s = 20,
               Nr_grid_t = 40, 
               time_from =0, 
               time_to = 1,
               minimize = True):
    # Initialize grids
    s = np.linspace(start_value_from,start_value_to,Nr_grid_s)
    Delta_s = s[1]-s[0]
    t = np.linspace(time_from,time_to,Nr_grid_t)
    Delta_t = t[1]-t[0]
    #print(1-(Delta_t/Delta_s**2)*max([a_0_upper_bound+a_1_upper_bound*val*(val>0) for val in s]))
    if Delta_s**2/max([a_0_upper_bound+a_1_upper_bound*val*(val>0) for val in s]) < Delta_t:
        print("WARNING: Stability condition is violated. \nChoose more time iterations!")
    u = np.zeros((len(t),len(s)))
    def G(x,p,q,minimize):
            possible_b = [(b_0_lower_bound+b_1_lower_bound*x)*p,
              (b_0_upper_bound+b_1_lower_bound*x)*p,
             (b_0_lower_bound+b_1_upper_bound*x)*p,
             (b_0_upper_bound+b_1_upper_bound*x)*p]
            possible_a = [(0.5*(a_0_lower_bound+a_1_lower_bound*x*(x>0))**(2*gamma_lower_bound))*q,
                      (0.5*(a_0_upper_bound+a_1_lower_bound*x*(x>0))**(2*gamma_lower_bound))*q,
                     (0.5*(a_0_lower_bound+a_1_upper_bound*x*(x>0))**(2*gamma_lower_bound))*q,
                     (0.5*(a_0_upper_bound+a_1_upper_bound*x*(x>0))**(2*gamma_lower_bound))*q,
                         (0.5*(a_0_lower_bound+a_1_lower_bound*x*(x>0))**(2*gamma_upper_bound))*q,
                      (0.5*(a_0_upper_bound+a_1_lower_bound*x*(x>0))**(2*gamma_upper_bound))*q,
                     (0.5*(a_0_lower_bound+a_1_upper_bound*x*(x>0))**(2*gamma_upper_bound))*q,
                     (0.5*(a_0_upper_bound+a_1_upper_bound*x*(x>0))**(2*gamma_upper_bound))*q]
            if minimize:
                return np.min(possible_b)+np.min(possible_a)
            else:
                return np.max(possible_b)+np.max(possible_a)
            
    #Boundary Condition for maturity T
    u[-1,:] = [payoff(val) for val in s]
    #Boundary Condition for border of space:
    u[:,0] = [payoff(s[0])]*len(t)
    u[:,-1] = [payoff(s[-1])]*len(t)
    time_step = 0
    # Iteration over the time steps
    for i in tqdm(range(len(t)-1,0,-1)):
        time_step += 1
        #Boundary Conditions for left and right side of the space grid
        #Left Boundary:
        # G_evaluated = G(s[0],
        #                 (1/(Delta_s))*(u[i,1]-u[i,0]),
        #                 (1/Delta_s**2)*(u[i,2]-2*u[i,1]+u[i,0]),
        #                 minimize)
        # u[i-1,0] = u[i,0]+G_evaluated*Delta_t
        # Right Boundary:
        # G_evaluated = G(s[-1],
        #               (1/(Delta_s))*(u[i,-1]-u[i,-2]),
        #               (1/Delta_s**2)*(u[i,-1]-2*u[i,-2]+u[i,-3]),
        #               minimize)
        # u[i-1,-1] = u[i,-1]+G_evaluated*Delta_t
        #Iteration over the inner values of the space
        for j in range(1,len(s)-1):
            G_evaluated = G(s[j],
                              (1/(2*Delta_s))*(u[i,j+1]-u[i,j-1]),
                              (1/Delta_s**2)*(u[i,j+1]-2*u[i,j]+u[i,j-1]),
                              minimize)
            u[i-1,j] = u[i,j]+G_evaluated*Delta_t
    return u[0,:], u