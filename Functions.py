# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:20:37 2021

@author: Julian
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import norm
from tqdm import tqdm

def generate_path(a_0, # Interval for Parameter a_0
                  a_1, # Interval for  Parameter a_1
                  b_0, # Interval for  Parameter b_0
                  b_1, # Interval for  Parameter b_1
                  gamma, # Interval for  Gamma
                  x_0, # Initial value
                  T, # Maturity (in years)
                  n, # Nr of trading days
                  seed = 0): 
    
        #Time difference between trading days:
        dt = T/n
        # Create the differences of a Brownian Motion
        # Set the random seed for the Brownian motion - if desired
        if seed != 0:
            np.random.seed(seed)    
        dW = np.sqrt(dt) * np.random.randn(n)
        # Apply the Euler Maruyama Scheme
        # Use real randomness for the parameters!
        np.random.seed()
        # Initial value
        X = [x_0]
        for i in range(n):
            # Choose random samples in each step
            a_0_sample = a_0[0] + (a_0[1]-a_0[0])*np.random.uniform() 
            a_1_sample = a_1[0] + (a_1[1]-a_1[0])*np.random.uniform()
            b_0_sample = b_0[0] + (b_0[1]-b_0[0])*np.random.uniform()
            b_1_sample = b_1[0] + (b_1[1]-b_1[0])*np.random.uniform()
            gamma_sample = gamma[0] + (gamma[1]-gamma[0])*np.random.uniform()
            #Compute the discretized value
            X += [X[-1]+(b_0_sample+b_1_sample*X[-1])*dt+(a_0_sample+a_1_sample*np.max([X[-1],0]))**gamma_sample*dW[i]]
        return X
  

# The corresponding generator
def generate_batch_of_paths(a_0, # Interval for Parameter a_0
                  a_1, # Interval for Parameter a_1
                  b_0, # Interval for Parameter b_0
                  b_1, # Interval for Parameter b_1
                  gamma, # Interval for  Gamma
                  x_0, # Initial value
                  T, # Maturity (in years)
                  n,
                  batch_size = 256,
                  scaling_factor = 1.): # Nr of trading days
    while True:
        batch = tf.reshape([generate_path(a_0,a_1,b_0,b_1,gamma, x_0,T,n) for i in range(batch_size)],([batch_size,n+1]))
        yield batch/scaling_factor
        


def optimal_hedge(derivative, # Function describing the payoff of the derivative to hedge
                  a_0, # Interval for Parameter a_0
                  a_1, # Interval for Parameter a_1
                  b_0, # Interval for Parameter b_0
                  b_1, # Interval for Parameter b_1
                  gamma, # Parameter gamma
                  x_0, # Initial value
                  T, # Maturity (in years)
                  n,# Nr of trading days
                 depth = 2, # Depth of the neural network (nr of hidden layers)
                 nr_neurons = 15, # Nr of neurons per layer
                EPOCHS = 1000, # Total number of epocs
                  l_r = 0.0001, # Learning rate of the Adam optimizer,
                  BATCH_SIZE =256, # Batch size for sampling the paths,
                  hedge = "hedge",
                  scaling_factor =1.,
                  path_dependent = False
                 ): 
    #x_0 = tf.cast(x_0,tf.float32)
    #List of Trading Days
    first_path = next(generate_batch_of_paths(a_0,a_1,b_0,b_1,gamma,x_0,T,n,BATCH_SIZE,scaling_factor))
    Initial_value = tf.reduce_mean(tf.map_fn(derivative,first_path))  
    t_k = np.linspace(0,T,n+1)
    alpha = tf.Variable([Initial_value],trainable=True,dtype = "float32")
    
    # Define the neural networks
    def build_model(depth,nr_neurons):
        if path_dependent:
            x = keras.Input(shape=(1,),name = "x")
            t = keras.Input(shape=(1,),name = "t")
            max_x = keras.Input(shape=(1,),name = "max_x")
            fully_connected_Input = layers.concatenate([x, t,max_x]) 
        else:            
            x = keras.Input(shape=(1,),name = "x")
            t = keras.Input(shape=(1,),name = "t")
            fully_connected_Input = layers.concatenate([x, t])         
        # Create the NN       
        values_all = layers.Dense(nr_neurons,activation = "relu")(fully_connected_Input)       
        # Create deep layers
        for i in range(depth):
            values_all = layers.Dense(nr_neurons,activation = "relu")(values_all)            
        # Output Layers
        value_out = layers.Dense(1)(values_all)
        if path_dependent:
            model = keras.Model(inputs=[x,t,max_x],
                    outputs = [value_out])
        else:
            model = keras.Model(inputs=[x,t],
                    outputs = [value_out])
        return model
    
    # Define Risk Measure    
    #def rho(x): # Inpur as a list of entries!, Entropy with lambda = 1
    #    return tf.math.log(tf.reduce_mean(tf.math.exp(-x)))
    
    if hedge == "hedge":    
        def rho(x):
            return tf.reduce_mean(tf.math.square(x))
    if hedge == "super-hedge":
        def rho(x):
            return tf.reduce_mean(tf.math.square(x))+tf.reduce_mean(tf.math.square(tf.nn.relu(-x)))
    if hedge == "sub-hedge":
        def rho(x):
            return tf.reduce_mean(tf.math.square(x))+tf.reduce_mean(tf.math.square(tf.nn.relu(x)))
        
    # Define the Loss function    
    def loss(model,batch):        
        #model_evaluated = [model([tf.reshape(batch[:,i],(BATCH_SIZE,1)),tf.reshape(np.repeat(t_k[i],BATCH_SIZE),(BATCH_SIZE,1))]) for i in range(n)]
        #delta_S = tf.reduce_sum([model_evaluated[i]*np.reshape(np.diff(batch)[:,i],(BATCH_SIZE,1)) for i in range(n)],0)
        #derivative_on_batch = np.array([[derivative(batch[i,:])] for i in range(BATCH_SIZE)])       
        patch_diff = batch[:,1:]-batch[:,:-1]
        if path_dependent:
            hedge_evaluated = [model([tf.reshape(batch[:,i],(BATCH_SIZE,1)),
                                      tf.reshape(np.repeat(t_k[i],BATCH_SIZE),(BATCH_SIZE,1)),
                                      tf.reshape(tf.reduce_max(batch[:,:(i+1)],1),(BATCH_SIZE,1))]) for i in range(n)]
        else:
            hedge_evaluated = [model([tf.reshape(batch[:,i],(BATCH_SIZE,1)),
                                      tf.reshape(np.repeat(t_k[i],BATCH_SIZE),(BATCH_SIZE,1))]) for i in range(n)]
        delta_S = tf.reduce_sum(tf.math.multiply(patch_diff,tf.transpose(tf.reshape(hedge_evaluated,(n,BATCH_SIZE)))),1)
        derivative_on_batch = tf.map_fn(derivative,batch)
        loss = rho(alpha+delta_S-derivative_on_batch)
        return loss
    
    # Define Gradient    
    def grad(model,batch):
        with tf.GradientTape() as tape:
            loss_value = loss(model,batch)
        return loss_value, tape.gradient(loss_value,model.trainable_variables+[alpha])

    def grad_alpha(model,batch):
        with tf.GradientTape() as tape:
            loss_value = loss(model,batch)
        return tape.gradient(loss_value,[alpha])
    
    # Create Optimizer and Model
    optimizer = tf.keras.optimizers.Adam(learning_rate = l_r, beta_1=0.9, beta_2=0.999)
    optimizer_alpha = tf.keras.optimizers.SGD(learning_rate = 10*l_r)
    model = build_model(depth,nr_neurons)
    losses = []

    # Training Loop
    for epoch in tqdm(range(int(EPOCHS))):
        batch = next(generate_batch_of_paths(a_0,a_1,b_0,b_1,gamma,x_0,T,n,BATCH_SIZE,scaling_factor))
        loss_value, grads = grad(model,batch)
        #grads_a =  grad_alpha(model,batch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables+[alpha]))
        #optimizer_alpha.apply_gradients(zip(grads_a,[alpha]))
        losses.append(loss_value.numpy()*scaling_factor)
        if epoch % 10 == 0 and epoch > 0:
            print("Iteration:{}, Price of Hedge: {}, Loss: {}".format((epoch),alpha.numpy()[0]*scaling_factor,losses[-1]))         
    return np.mean(alpha.numpy()[0]), model

def BS_Delta(sigma,r,S_0,T,K):
    d_1 = (np.log(S_0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d_1)

def BS_Price(sigma,r,S_0,T,K,Call_Put):
    d_1 = (np.log(S_0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d_2 = d_1 -sigma*np.sqrt(T)
    if Call_Put == "Call":
        return S_0*norm.cdf(d_1)-K*np.exp(-r*T)*norm.cdf(d_2)
    elif Call_Put == "Put":
        return K*np.exp(-r*T)*norm.cdf(-d_2)-S_0*norm.cdf(-d_1)
        
    