from __future__ import division
#import numpy as np

import autograd.numpy as np
from autograd import grad, jacobian, hessian

import matplotlib.pyplot as plt
import argparse
from scipy.stats import gamma
from scipy.optimize import minimize,fmin_l_bfgs_b

def measure_sensitivity(X):
    N = len(X)
    Ds = 1/N * (np.abs(np.max(X) - np.min(X)))

    return(Ds)
    
def measure_sensitivity_private(distribution, N, theta_vector):
    #computed on a different sample than the one being analyzed
    if distribution == 'poisson':
        theta = theta_vector[0]
        Xprime = np.random.poisson(theta, size=N)
        Xmax, Xmin = np.max(Xprime), np.min(Xprime)
        Ds = 1/N * (np.abs(Xmax - Xmin))

        
    if distribution == 'gaussian':

        theta, sigma = theta_vector[0], theta_vector[1]
        Xprime = np.random.normal(theta, sigma, size=N)
        Xmax, Xmin = np.max(Xprime), np.min(Xprime)
        Ds = 1/N * (np.abs(Xmax - Xmin))

        
    if distribution == 'gaussian2':

        theta, sigma = theta_vector[0], theta_vector[1]
        Xprime = np.random.normal(theta, sigma, size=N)
        Xmax, Xmin = np.max(Xprime), np.min(Xprime)
        Ds1 = 1/N * (np.abs(Xmax - Xmin))
        #Xmax2, Xmin2 = np.max(Xprime**2), np.min(Xprime**2)
        #Ds2 = 1/N * 1/N * ((Xmax - Xmin)**2)    # see du et al 2020 Th 26 following honacker
        Ds2 = 1/N * ((Xmax - Xmin)**2)
        
        Ds = np.abs(Ds1) + np.abs(Ds2)

        
    if distribution == 'gamma':
        theta, theta2 = theta_vector[0], theta_vector[1]
        Xprime = np.random.gamma(theta2, theta, size=N)
        Xmax, Xmin = np.max(Xprime), np.min(Xprime)
        Ds = 1/N * (np.abs(Xmax - Xmin))

        
    if distribution == 'gaussianMV':
        theta, theta2 = theta_vector[0], theta_vector[1]
        K = len(theta)
        Xprime = np.random.multivariate_normal(theta, theta2, size=N)
        Xmax, Xmin = np.max(Xprime, axis=0), np.min(Xprime,axis=0)
        Ds = 1/N * (np.abs(Xmax.T-Xmin.T))
        
    return(Ds, [Xmin, Xmax])
    
        
def A_SSP(X, Xdistribution, privately_computed_Ds, laplace_noise_scale, theta_vector):

    N = len(X)
    
    if Xdistribution == 'poisson':

        s = 1/N * np.sum(X)
        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale, size = 1)

        theta_hat_given_s = s
        theta_hat_given_z = z
        
        return({'0priv': theta_hat_given_z, '0basic': theta_hat_given_s})
        
    if Xdistribution == 'gaussian':

        s = 1/N * np.sum(X)

        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale, size = 1)

        theta_hat_given_s = s
        theta_hat_given_z = z
        
        return({'0priv': theta_hat_given_z, '0basic': theta_hat_given_s})

        s1 = 1/N * np.sum(X)
        s2 = 1/(N-1) * np.sum((X-s1)**2)

        z1 = np.random.laplace(loc=s1, scale=privately_computed_Ds/(laplace_noise_scale*0.5), size = 1)
        
        negative = True
        
        while negative:
            z2 = np.random.laplace(loc=s2, scale=privately_computed_Ds/(laplace_noise_scale*0.5), size = 1)
            if z2>0:
                negative = False
        
        theta_hat_given_s = s1

        theta2_hat_given_s = np.sqrt(s2)

        #theta2_hat_given_s = np.sqrt(1/(N-1) * (N*s2 - N*(s1**2))) 
        theta_hat_given_z = z1
        theta2_hat_given_z = np.sqrt(z2)
        
        # print(theta_hat_given_s)
        # print(theta_hat_given_z)
        # print(theta2_hat_given_s)
        # print(theta2_hat_given_z)
        # print(stop)

        return({'0priv': theta_hat_given_z, '1priv': theta2_hat_given_z, '0basic': theta_hat_given_s, '1basic': theta2_hat_given_s})
    
    if Xdistribution == 'gamma':

        K = theta_vector[1]
        s = 1/N * np.sum(X)
        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale, size = 1)

        theta_hat_given_s = 1/K * s
        theta_hat_given_z = 1/K * z
        
        return({'1priv': theta_hat_given_z, '1basic': theta_hat_given_s})
        
    if Xdistribution == 'gaussianMV':
        mu = theta_vector[0]
        Sigma = theta_vector[1]
        s = 1/N * np.sum(X, axis=0)
        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale)
        
        theta_hat_given_s = s
        theta_hat_given_z = z
        
        return({'1priv': theta_hat_given_z, '1basic': theta_hat_given_s})
        
def A_SSP_autodiff(X, Xdistribution, privately_computed_Ds, laplace_noise_scale, theta_vector):
    
    N = len(X)
    
    theta_init = [1.9, 3.9]
    
    if Xdistribution == 'poisson':

        s = 1/N * np.sum(X)
        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale, size = 1)

        ll_jac = jacobian(negativeloglikelihood)
        theta_hat_given_s = minimize(negativeloglikelihood, theta_init, args=(Xdistribution, [s], N), method = 'BFGS', options={'gtol':1e-7,'disp': False}, jac = ll_jac).x[0]
        theta_hat_given_z = minimize(negativeloglikelihood, theta_init, args=(Xdistribution, z, N), method = 'BFGS', options={'gtol':1e-7,'disp': False}, jac = ll_jac).x[0]
        
        return({'0priv': theta_hat_given_z, '0basic': theta_hat_given_s})
        
    if Xdistribution == 'gaussian':

        s = 1/N * np.sum(X)
        
        
        z = np.random.laplace(loc=s, scale=privately_computed_Ds/laplace_noise_scale, size = 1)

        theta_hat_given_s = s
        theta_hat_given_z = z
        
        return({'0priv': theta_hat_given_z, '0basic': theta_hat_given_s})
        
def fisherInfo(d, N, params):
    
    if d == 'poisson':
        fishInf = N/params[0]

    elif d == 'gaussian':
        fishInf = N/(params[1]**2)
        
    elif d == 'gaussian2':
        fishInf = np.array([[N/(params[1]**2),0],[0,N/(2*params[1]**4)]])    
        
    elif d == 'gamma':
        K = params[1]
        scale = params[0]
        fishInf = N*K/(scale**2)
    
    elif d == 'gaussianMV':
        Sigma = params[1]
        fishInf = N*(np.linalg.inv(Sigma))
        
    return(fishInf)
        
# functions for numerical optimization
        
def negativeloglikelihood(params, d, suffstat, N):
    #https://en.wikipedia.org/wiki/Poisson_distribution#Parameter_estimation
    if d == 'poisson':
        theta = params[0]
        eta = np.log(theta)
        Tx = N * suffstat[0]
        A = theta
    if d == 'gaussian':
        theta = params[0]
        sigma = params[1]
        eta = theta/(sigma**2)
        Tx = N * suffstat[0]
        A = (theta**2)/(2*sigma**2)
    ll = eta * Tx - N * A
    return(-1*ll)   #didnt include constant
        
def optimization(suffstat, N):
    init = 0.5
    out = minimize(negativeloglikelihood, init, args=(suffstat, N), method='BFGS', jac=False, tol=1e-08,  options={'disp': False})
    return(out.x)

        