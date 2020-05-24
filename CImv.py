from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gamma
from scipy.optimize import minimize,fmin_l_bfgs_b

from functions import *


def parametricBootstrap(distribution, theta_vector, B, sensitivity, noise_scale, clipmin, clipmax, clip):
    
    [theta, theta2] = theta_vector

    if distribution == 'poisson':
        
        X = np.random.poisson(theta, N)
        
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
            
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0basic']
        
        #quick fix to avoid negative theta:
        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_priv = 0.00000001
        
        #bootstraps
        Xbs = np.random.poisson(theta_priv, (N,B))
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.poisson(theta_priv, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.poisson(theta_basic, (N,B)), axis = 0)
        
        #fishInf
        fishInf = fisherInfo(distribution, X, [theta_priv, theta2])

    if distribution == 'gaussian':
        X = np.random.normal(theta, theta2, N)
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0basic']

        
        #quick fix to avoid negative theta:
        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_priv = 0.00000001

        #bootstraps
        Xbs = np.random.normal(theta_priv, theta2, (N,B))     #variance of normal is assumed known, inference of theta 
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.normal(theta_priv, theta2, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.normal(theta_basic, theta2, (N,B)), axis = 0) 
        
        #fishInf
        fishInf = fisherInfo(distribution, X, [theta_priv, theta2])
        
    if distribution == 'gaussian2':
        X = np.random.normal(theta, theta2, N)
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0priv']
        theta2_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['0basic']
        theta2_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1basic']
        
        #quick fix to avoid negative theta:
        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_priv = 0.00000001

        #bootstraps
        Xbs = np.random.normal(theta_priv, theta2, (N,B))     #variance of normal is assumed known, inference of theta 
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.normal(theta_priv, theta2, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.normal(theta_basic, theta2, (N,B)), axis = 0) 
        
        #fishInf
        fishInf = fisherInfo(distribution, X, [theta_priv, theta2])
        
    if distribution == 'gamma':
        
        X = np.random.gamma(theta2, theta, N)
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1basic']

        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_priv = 0.00000001

        #bootstraps
        Xbs = np.random.gamma(theta2, theta_priv, (N,B))     #theta known, inference on theta2
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = 1/theta2 * np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/theta2 * 1/N * np.sum(np.random.gamma(theta2, theta_priv, (N,B)), axis = 0) 
        theta_tildas_basic = 1/theta2 * 1/N * np.sum(np.random.gamma(theta2, theta_basic, (N,B)), axis = 0) 
        
        #fishInf
        fishInf = fisherInfo(distribution, X, [theta_priv, theta2])
        
    
    if distribution == 'gaussianMV':
        X = np.random.multivariate_normal(theta, theta2, size=N, check_valid = 'warn')
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector)['1basic']
        
        #bootstraps
        Xbs = np.random.multivariate_normal(theta_priv, theta2, size=(B,N), check_valid = 'warn')  #theta known, inference on theta2
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=1), scale = np.array(sensitivity)/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.multivariate_normal(theta_priv, theta2, size=(B,N), check_valid = 'warn'), axis = 1) 
        theta_tildas_basic =1/N * np.sum(np.random.multivariate_normal(theta_basic, theta2, size=(B,N), check_valid = 'warn'), axis = 1) 
        
        #fishInf
        fishInf = fisherInfo(distribution, X, [theta_priv, theta2])
        
        
    return(theta_tildas, theta_tildas_naive, theta_tildas_basic, fishInf, theta_priv)


    
########################################################################################################################
# CI EXPERIMENT
########################################################################################################################    
    
def CIs(distribution, theta_vector, N, B, noise_scale, mode, T, clip):
    
    # get params
    [theta, theta2] = theta_vector
    
    K = len(theta)

    # compute sensitivity pre-privatization
    sensitivity, [clipmin, clipmax] = measure_sensitivity_private(distribution, N, theta_vector)
    
    # be cautious and multiply it by 2 (but later we'll likely change this and can clip values outside bounds
    if not clip: sensitivity = sensitivity*2

    ############## PRIVACY BOUNDARY ####################################################################################
    # no access to X from here on
    
    # start bootstrap experiment to find CIs

    # init data storage
    results, results_naive, results_basic, results_fisher = [], [], [], []
    
    list_upper_failures, list_lower_failures = [],[]
    list_upper_failures_naive, list_lower_failures_naive = [],[]
    list_upper_failures_basic, list_lower_failures_basic  = [],[]
    list_upper_failures_fisher, list_lower_failures_fisher = [],[]
    
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    list_ci_levels = sorted(list(z_values.keys()))
      
    for coverage in list_ci_levels:
           
        trial_results = np.zeros((T,3,K))
        trial_results_naive = np.zeros((T,3,K))
        trial_results_basic = np.zeros((T,3,K))
        trial_results_fisher = np.zeros((T,3,K))
        
        num_upper_failures, num_lower_failures = 0, 0 
        num_upper_failures_naive, num_lower_failures_naive = 0, 0
        num_upper_failures_basic, num_lower_failures_basic = 0, 0
        num_upper_failures_fisher, num_lower_failures_fisher = 0, 0
        
        
        #run T confidence interval trials
            
        for t in range(T):
            
            theta_tildas, theta_tildas_naive, theta_tildas_basic, fishInf, theta_priv = parametricBootstrap(distribution, theta_vector, B, sensitivity, noise_scale, clipmin, clipmax, clip)
            # bootstrap completed, now find statistics based on the theta tilde vectors found via bootstrap
            mu = np.mean(theta_tildas, axis=0)
            std = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas,mu))**2, axis=0))
            
            mu_naive = np.mean(theta_tildas_naive, axis=0)
            std_naive = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas_naive,mu_naive))**2, axis=0))
            
            mu_basic = np.mean(theta_tildas_basic, axis=0)
            std_basic = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas_basic,mu_basic))**2, axis=0))
            
            S = np.linalg.inv(fishInf)
            
            trial_results[t,1,:] = mu
            trial_results[t,2,:] = std
            
            trial_results_naive[t,1,:] = mu_naive
            trial_results_naive[t,2,:] = std_naive
            
            trial_results_basic[t,1,:] = mu_basic
            trial_results_basic[t,2,:] = std_basic
            
            trial_results_fisher[t,1,:] = theta_priv
            trial_results_fisher[t,2,:] = S[0,:]
            
            
            # find confidence interval bounds
            
            if mode=='analytic': #this is the standard normal interval method (see Efron-Tibshirani)
            
                CI_lower = mu[0] - z_values[coverage]*std[0] 
                CI_upper = mu[0] + z_values[coverage]*std[0]

                if theta[0] >= CI_lower and theta[0] <= CI_upper:
                    trial_results[t,0]=1.0
                else:
                    if theta[0] < CI_lower: num_lower_failures += 1
                    elif theta[0] > CI_upper: num_upper_failures += 1
                    
                CI_lower_naive = mu_naive[0] - z_values[coverage]*std_naive[0]
                CI_upper_naive = mu_naive[0] + z_values[coverage]*std_naive[0]
                if theta[0] >= CI_lower_naive and theta[0] <= CI_upper_naive:
                    trial_results_naive[t,0]=1.0
                else:
                    if theta[0] < CI_lower_naive: num_lower_failures_naive += 1
                    elif theta[0] > CI_upper_naive: num_upper_failures_naive += 1
                    
                CI_lower_basic = mu_basic[0] - z_values[coverage]*std_basic[0]
                CI_upper_basic = mu_basic[0] + z_values[coverage]*std_basic[0]
                if theta[0] >= CI_lower_basic and theta[0] <= CI_upper_basic:
                    trial_results_basic[t,0]=1.0
                else:
                    if theta[0] < CI_lower_basic: num_lower_failures_basic += 1
                    elif theta[0] > CI_upper_basic: num_upper_failures_basic += 1
                    
                # FISHER INFO

                CI_lower_fish = theta_priv[0] - z_values[coverage]*(np.sqrt(S[0,0]))
                CI_upper_fish = theta_priv[0] + z_values[coverage]*(np.sqrt(S[0,0]))
                if theta[0] >= CI_lower_fish and theta[0] <= CI_upper_fish:
                    trial_results_fisher[t,0]=1.0
                else:
                    if theta[0] < CI_lower_fish: num_lower_failures_fisher += 1
                    elif theta[0] > CI_upper_fish: num_upper_failures_fisher += 1
            
                    
            
            elif mode=='empirical':  #boostrap CIs
                conf_level = coverage
                alpha = 100-conf_level
                
                CI_upper= np.percentile(theta_tildas[:,0], 100-alpha/2.0)
                CI_lower= np.percentile(theta_tildas[:,0], alpha/2.0)
    
                if theta[0]>= CI_lower and theta[0]<=CI_upper:
                    trial_results[t,0]=1.0
                else:
                    if theta[0] < CI_lower: num_lower_failures += 1
                    elif theta[0] > CI_upper: num_upper_failures += 1
                    
                CI_upper_naive = np.percentile(theta_tildas_naive[:,0], 100-alpha/2.0)
                CI_lower_naive = np.percentile(theta_tildas_naive[:,0], alpha/2.0)
    
                if theta[0] >= CI_lower_naive and theta[0] <= CI_upper_naive:
                    trial_results_naive[t,0]=1.0
                else:
                    if theta[0] < CI_lower_naive: num_lower_failures_naive += 1
                    elif theta[0] > CI_upper_naive: num_upper_failures_naive += 1
                    
                CI_upper_basic = np.percentile(theta_tildas_basic[:,0], 100-alpha/2.0)
                CI_lower_basic = np.percentile(theta_tildas_basic[:,0], alpha/2.0)
    
                if theta[0] >= CI_lower_basic and theta[0] <= CI_upper_basic:
                    trial_results_basic[t,0]=1.0
                else:
                    if theta[0] < CI_lower_basic: num_lower_failures_basic += 1
                    elif theta[0] > CI_upper_basic: num_upper_failures_basic += 1
                    
                # FISHER INFO
                
                CI_lower_fish = theta_priv[0] - z_values[coverage]*(np.sqrt(S[0,0]))
                CI_upper_fish = theta_priv[0] + z_values[coverage]*(np.sqrt(S[0,0]))
                if theta >= CI_lower_fish and theta <= CI_upper_fish:
                    trial_results_fisher[t,0]=1.0
                else:
                    if theta[0] < CI_lower_fish: num_lower_failures_fisher += 1
                    elif theta[0] > CI_upper_fish: num_upper_failures_fisher += 1
        
                    
                    
        # store results
        results.append(np.mean(trial_results, axis=0))
        results_naive.append(np.mean(trial_results_naive, axis=0))
        results_basic.append(np.mean(trial_results_basic, axis=0))
        results_fisher.append(np.mean(trial_results_fisher, axis=0))
        
        list_upper_failures.append(num_upper_failures)
        list_lower_failures.append(num_lower_failures)
        list_upper_failures_naive.append(num_upper_failures_naive)
        list_lower_failures_naive.append(num_lower_failures_naive)
        list_upper_failures_basic.append(num_upper_failures_basic)
        list_lower_failures_basic.append(num_lower_failures_basic)
        list_upper_failures_fisher.append(num_upper_failures_fisher)
        list_lower_failures_fisher.append(num_lower_failures_fisher)
    
    print(results)
    print("\n")
    #print(results_naive)
    print(results_fisher)
    print("\n")
    #print(results_basic[0,:])
    print("\n")
    
    #save results

    name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '.npy'
    #np.save('coveragelevels_' + name_suffix, list_ci_levels)
    np.save('results_' + name_suffix, results)
    np.save('upperfailures_' + name_suffix, list_upper_failures)
    np.save('lowerfailures_' + name_suffix, list_lower_failures)
    
    name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '_NAIVE.npy'
    #np.save('coveragelevels_' + name_suffix, list_ci_levels)
    np.save('results_' + name_suffix, results_naive)
    np.save('upperfailures_' + name_suffix, list_upper_failures_naive)
    np.save('lowerfailures_' + name_suffix, list_lower_failures_naive)
    
    name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '_BASIC.npy'
    #np.save('coveragelevels_' + name_suffix, list_ci_levels)
    np.save('results_' + name_suffix, results_basic)
    np.save('upperfailures_' + name_suffix, list_upper_failures_basic)
    np.save('lowerfailures_' + name_suffix, list_lower_failures_basic)
    
    name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '_FISHER.npy'
    #np.save('coveragelevels_' + name_suffix, list_ci_levels)
    np.save('results_' + name_suffix, results_fisher)
    np.save('upperfailures_' + name_suffix, list_upper_failures_fisher)
    np.save('lowerfailures_' + name_suffix, list_lower_failures_fisher)
        
    
    
if __name__ == "__main__":
    
    np.random.seed(22)
    
    parser = argparse.ArgumentParser(description='Confidence Intervals for Private Estimators')
    
    parser.add_argument('--d', type=str, default='gaussianMV', help='distribution (poisson, gaussian, gamma)')
    parser.add_argument('--mode', type=str, default='analytic', help='analytic or empirical (CI mode)')
    parser.add_argument('--e', type=float, default=0.5, help='DP epsilon')
    parser.add_argument('--clip', type=bool, default=True, help='clip data outside bounds')
    
    parser.print_help()
    
    args = parser.parse_args()
    
    if args.d in ['gaussian', 'poisson', 'gamma', 'gaussian2']:
        theta = np.random.rand() * 20
        theta2 = np.random.rand() * 8
        
    else:
        k = 5
        theta = np.array([np.random.rand() * 20 for i in range(k)])
        theta2 = np.array([[np.random.rand()* 3 for i in range(k)] for j in range(k)])
        theta2 = np.dot(theta2, theta2.T)
        
    
    for N in [50,100,200,500,1000,5000,10000]:
        CIs(args.d, [theta, theta2], N, 500, args.e, args.mode, 500)
