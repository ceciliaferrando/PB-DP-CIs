from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gamma
from scipy.optimize import minimize,fmin_l_bfgs_b
import pickle

from functions import *


def parametricBootstrap(distribution, N, theta_vector, B, sensitivity, noise_scale, clipmin, clipmax, clip, coverage, rho):
    
    [theta, theta2] = theta_vector

    if distribution == 'poisson':
        
        X = np.random.poisson(theta, N)
        
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
            
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0basic']
        
        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_basic = 0.00000001
        
        #bootstraps
        Xbs = np.random.poisson(theta_priv, (N,B))
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.poisson(theta_priv, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.poisson(theta_basic, (N,B)), axis = 0)

        theta_tildas[theta_tildas<=0] = 0.00000001
        theta_tildas_naive[theta_tildas_naive<=0] = 0.00000001
        theta_tildas_basic[theta_tildas_basic<=0] = 0.00000001

        # bias quantification
        e_theta_tilda = np.mean(theta_tildas)
        estimated_bias = e_theta_tilda - theta_priv
        true_bias = theta_priv - theta

        #fisher information for the fisher CI
        fishInf = fisherInfo(distribution, N, [theta_priv, theta2])
        fishInfInvSqrt = fishInf**(-1/2)
        fishInfInvSqrtTildeVec = [fisherInfo(distribution, N, [theta_tilda, theta2]) ** (-1 / 2) for theta_tilda in
                                  theta_tildas]
        
        var_w = 2 * (sensitivity/noise_scale)**2
        fishInfInvSqrtCorr = np.sqrt(fishInf**(-1) + fishInf**(-2)*var_w)
        
        fishInfNP = fisherInfo(distribution, N, [theta_basic, theta2])
        fishInfInvSqrtNP = fishInfNP**(-1/2)

    if distribution == 'gaussian':
        X = np.random.normal(theta, theta2, N)
        
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
            
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0basic']

        #bootstraps
        Xbs = np.random.normal(theta_priv, theta2, (N,B))     #variance of normal is assumed known, inference of theta 
        
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
            
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/N * np.sum(np.random.normal(theta_priv, theta2, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.normal(theta_basic, theta2, (N,B)), axis = 0)

        # bias quantification
        e_theta_tilda = np.mean(theta_tildas)
        estimated_bias = e_theta_tilda - theta_priv
        
        #fisher information for the fisher CI
        fishInf = fisherInfo(distribution, N, [theta_priv, theta2])
        fishInfInvSqrt = fishInf**(-1/2)
        fishInfInvSqrtTildeVec = [fisherInfo(distribution, N, [theta_tilda, theta2])**(-1/2) for theta_tilda in theta_tildas]

        
        var_w = 2 * (sensitivity/noise_scale)**2
        fishInfInvSqrtCorr = np.sqrt(fishInf**(-1) + fishInf**(-2)*var_w)
        
        fishInfNP = fisherInfo(distribution, N, [theta_basic, theta2])
        fishInfInvSqrtNP = fishInfNP**(-1/2)
        
    if distribution == 'gaussian2':
        
        CI = None
        
        X = np.random.normal(theta, theta2, N)
        
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
            
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0priv']
        theta2_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['1priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['0basic']
        theta2_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['1basic']

        if theta2_priv < 0: theta2_priv = 0.00000001
        if theta2_basic < 0: theta2_basic = 0.00000001

        #bootstraps
        Xbs = np.random.normal(theta_priv, theta2_priv, (N,B))    #inference on both
        
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
            
        theta_tildas = np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity[0]/(noise_scale*rho))
        theta_tildas_naive = 1/N * np.sum(np.random.normal(theta_priv, theta2_priv, (N,B)), axis = 0) 
        theta_tildas_basic = 1/N * np.sum(np.random.normal(theta_basic, theta2_basic, (N,B)), axis = 0)

        #bias quantification
        estimated_bias = None   #to develop

        #fisher information for the fisher CI
        fishInf = fisherInfo(distribution, N, [theta_priv, theta2_priv])
        fishInfInvSqrt = np.sqrt(np.linalg.inv(fishInf)[0,0])
        fishInfInvSqrtTildeVec = [np.sqrt(np.linalg.inv(fisherInfo(distribution, N, [theta_tilda, theta2_priv]))[0,0]) ** (-1 / 2)
                                  for theta_tilda in theta_tildas]
        print(fishInfInvSqrtTildeVec)
        print(stop)
        
        #var_w = 2 * (sensitivity[0]/noise_scale)**2 
        fishInfInvSqrtCorr = fishInfInvSqrt
        
        fishInfNP = fisherInfo(distribution, N, [theta_basic, theta2_basic])
        fishInfInvSqrtNP = np.sqrt(np.linalg.inv(fishInfNP)[0,0])  
        
    if distribution == 'gamma':
        
        X = np.random.gamma(theta2, theta, N)
        
        if clip:
            X[X<clipmin] = clipmin
            X[X>clipmax] = clipmax
            
        theta_priv = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['1priv']
        theta_basic = A_SSP(X, distribution, sensitivity, noise_scale, theta_vector, rho)['1basic']

        if theta_priv < 0: theta_priv = 0.00000001
        if theta_basic < 0: theta_basic = 0.00000001

        #bootstraps
        Xbs = np.random.gamma(theta2, theta_priv, (N,B))     #theta known, inference on theta2
        if clip:
            Xbs[Xbs<clipmin] = clipmin
            Xbs[Xbs>clipmax] = clipmax
        theta_tildas = 1/theta2 * np.random.laplace(loc = 1/N * np.sum(Xbs, axis=0), scale = sensitivity/noise_scale)
        theta_tildas_naive = 1/theta2 * 1/N * np.sum(np.random.gamma(theta2, theta_priv, (N,B)), axis = 0) 
        theta_tildas_basic = 1/theta2 * 1/N * np.sum(np.random.gamma(theta2, theta_basic, (N,B)), axis = 0) 

        theta_tildas[theta_tildas<=0] = 0.00000001
        theta_tildas_naive[theta_tildas_naive<=0] = 0.00000001
        theta_tildas_basic[theta_tildas_basic<=0] = 0.00000001

        #bias quantification
        e_theta_tilda = np.mean(theta_tildas)
        estimated_bias = e_theta_tilda - theta_priv
        
        #fisher information for the fisher CI
        fishInf = fisherInfo(distribution, N, [theta_priv, theta2])
        fishInfInvSqrt = fishInf**(-1/2)
        fishInfInvSqrtTildeVec = [fisherInfo(distribution, N, [theta_tilda, theta2]) ** (-1 / 2) for theta_tilda in
                                  theta_tildas]
        
        var_w = 2 * (sensitivity/noise_scale)**2
        fishInfInvSqrtCorr = np.sqrt(fishInf**(-1) + fishInf**(-2)*var_w)
        
        fishInfNP = fisherInfo(distribution, N, [theta_basic, theta2])
        fishInfInvSqrtNP = fishInfNP**(-1/2)
        
    return(theta_tildas, theta_tildas_naive, theta_tildas_basic, fishInfInvSqrt, fishInfInvSqrtTildeVec, fishInfInvSqrtCorr,
           fishInfInvSqrtNP, theta_priv, theta_basic, estimated_bias)


    
########################################################################################################################
# CI EXPERIMENT
########################################################################################################################    
    
def CIs(distribution, theta_vector, N, B, noise_scale, mode, T, cliplo, cliphi, rho):
    
    # get params
    [theta, theta2] = theta_vector
    print("param 1", theta)
    print("param 2 (if any)", theta2)

    # compute sensitivity pre-privatization
    sensitivity, [clipmin, clipmax] = measure_sensitivity_private(distribution, N, theta_vector, cliplo, cliphi)

    #clipmax = 10

    ############## PRIVACY BOUNDARY ####################################################################################
    # no access to X from here on
    
    # start bootstrap experiment to find CIs

    # init data storage
    results, results_db, results_naive, results_basic, results_fisher, results_fisher_corr, results_fisher_np = [], [], [], [], [], [], []
    widths, widths_db, widths_FI, widths_FI_np = [], [], [], []
    list_upper_failures, list_lower_failures = [],[]
    list_upper_failures_db, list_lower_failures_db = [], []
    list_upper_failures_naive, list_lower_failures_naive = [],[]
    list_upper_failures_basic, list_lower_failures_basic  = [],[]
    list_upper_failures_fisher, list_lower_failures_fisher = [],[]
    list_upper_failures_fisher_corr, list_lower_failures_fisher_corr = [],[]
    list_upper_failures_fisher_np, list_lower_failures_fisher_np = [],[]
    
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    list_ci_levels = sorted(list(z_values.keys()))
      
    for coverage in list_ci_levels:
           
        trial_results = np.zeros((T,3,))
        trial_results_db = np.zeros((T, 3,))
        trial_results_naive = np.zeros((T,3,))
        trial_results_basic = np.zeros((T,3,))
        trial_results_fisher = np.zeros((T,3,))
        trial_results_fisher_np = np.zeros((T,3,))
        trial_results_fisher_corr = np.zeros((T,3,))
        trial_errors = []
        Rs = np.zeros((T,2))
        Rs_db = np.zeros((T, 2))
        Rs_FI = np.zeros((T,2))
        Rs_FI_np = np.zeros((T,2))
        trial_estimated_bias = np.zeros((T,))
        
        
        num_upper_failures, num_lower_failures = 0, 0
        num_upper_failures_db, num_lower_failures_db = 0, 0
        num_upper_failures_naive, num_lower_failures_naive = 0, 0
        num_upper_failures_basic, num_lower_failures_basic = 0, 0
        num_upper_failures_fisher, num_lower_failures_fisher = 0, 0
        num_upper_failures_fisher_np, num_lower_failures_fisher_np = 0, 0
        num_upper_failures_fisher_corr, num_lower_failures_fisher_corr = 0, 0
        
        
        #run T confidence interval trials
            
        for t in range(T):
            clip = True
            (theta_tildas, theta_tildas_naive, theta_tildas_basic, fishInfInvSqrt, fishInfInvSqrtTildeVec,
             fishInfInvSqrtCorr, fishInfInvSqrtNP, theta_priv,
             theta_basic, estimated_bias) = parametricBootstrap(distribution, N, theta_vector, B,
                                                                           sensitivity, noise_scale, clipmin, clipmax,
                                                                           clip, coverage, rho)

            # bootstrap completed, now find statistics based on the bootstrap vectors

            theta_tildas_db = theta_tildas - estimated_bias
            
            mu = np.mean(theta_tildas)
            std = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas,mu))**2))

            mu_db = np.mean(theta_tildas_db)
            std_db = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas_db, mu)) ** 2))
            
            mu_naive = np.mean(theta_tildas_naive)
            std_naive = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas_naive,mu_naive))**2))
            
            mu_basic = np.mean(theta_tildas_basic)
            std_basic = np.sqrt(np.mean(np.abs(np.subtract(theta_tildas_basic,mu_basic))**2))
            
            trial_results[t,1] = mu
            trial_results[t,2] = std

            trial_results_db[t,1] = mu_db
            trial_results_db[t,2] = std_db
            
            trial_results_naive[t,1] = mu_naive
            trial_results_naive[t,2] = std_naive
            
            trial_results_basic[t,1] = mu_basic
            trial_results_basic[t,2] = std_basic
            
            trial_results_fisher[t,1] = theta_priv
            trial_results_fisher[t,2] = fishInfInvSqrt
            
            trial_results_fisher_corr[t,1] = theta_priv
            trial_results_fisher_corr[t,2] = fishInfInvSqrtCorr
            
            trial_results_fisher_np[t,1] = theta_basic
            trial_results_fisher_np[t,2] = fishInfInvSqrtNP

            trial_estimated_bias[t] = estimated_bias

            
            # find confidence interval bounds
            
            # FISHER INFO
            CI_lower_fish = theta_priv - z_values[coverage]*(fishInfInvSqrt)
            CI_upper_fish = theta_priv + z_values[coverage]*(fishInfInvSqrt)
            if CI_lower_fish<=0 and distribution in ['poisson', 'gamma']: CI_lower_fish = 0.00000001

            if theta >= CI_lower_fish and theta <= CI_upper_fish:
                trial_results_fisher[t,0]=1.0
            else:
                if theta < CI_lower_fish: num_lower_failures_fisher += 1
                elif theta > CI_upper_fish: num_upper_failures_fisher += 1

            R_FI = np.array([CI_lower_fish, CI_upper_fish]).T
            Rs_FI[t,:] = R_FI
                
            # FISHER INFO CORRECTED
            error = (theta_priv - theta)**2
            trial_errors.append([error])
            CI_lower_fish_corr = theta_priv - z_values[coverage]*(fishInfInvSqrtCorr)
            CI_upper_fish_corr = theta_priv + z_values[coverage]*(fishInfInvSqrtCorr)
            if CI_lower_fish_corr<=0 and distribution in ['poisson', 'gamma']: CI_lower_fish_corr = 0.00000001
            if theta >= CI_lower_fish_corr and theta <= CI_upper_fish_corr:
                trial_results_fisher_corr[t,0]=1.0
            else:
                if theta < CI_lower_fish_corr: num_lower_failures_fisher_corr += 1
                elif theta > CI_upper_fish_corr: num_upper_failures_fisher_corr += 1
                
            # FISHER INFO PUBLIC
            CI_lower_fish_np = theta_basic - z_values[coverage]*(fishInfInvSqrtNP)
            CI_upper_fish_np = theta_basic + z_values[coverage]*(fishInfInvSqrtNP)
            if CI_lower_fish_np <=0 and distribution in ['poisson', 'gamma']: CI_lower_fish_np = 0.00000001
            if theta >= CI_lower_fish_np and theta <= CI_upper_fish_np:
                trial_results_fisher_np[t,0]=1.0
            else:
                if theta < CI_lower_fish_np: num_lower_failures_fisher_np += 1
                elif theta > CI_upper_fish_np: num_upper_failures_fisher_np += 1
            R_FI_np = np.array([CI_lower_fish_np, CI_upper_fish_np]).T
            Rs_FI_np[t,:] = R_FI_np
                

            if mode=='analytic': #this is the standard normal interval method (see Efron-Tibshirani)
            
                CI_lower = mu - z_values[coverage]*std 
                CI_upper = mu + z_values[coverage]*std
                if CI_lower<=0 and distribution in ['poisson', 'gamma']: CI_lower = 0.00000001

                if theta >= CI_lower and theta <= CI_upper:
                    trial_results[t,0]=1.0
                else:
                    if theta < CI_lower: num_lower_failures += 1
                    elif theta > CI_upper: num_upper_failures += 1
                    
                CI_lower_naive = mu_naive - z_values[coverage]*std_naive
                CI_upper_naive = mu_naive + z_values[coverage]*std_naive
                if theta >= CI_lower_naive and theta <= CI_upper_naive:
                    trial_results_naive[t,0]=1.0
                else:
                    if theta < CI_lower_naive: num_lower_failures_naive += 1
                    elif theta > CI_upper_naive: num_upper_failures_naive += 1
                    
                CI_lower_basic = mu_basic - z_values[coverage]*std_basic
                CI_upper_basic = mu_basic + z_values[coverage]*std_basic
                if theta >= CI_lower_basic and theta <= CI_upper_basic:
                    trial_results_basic[t,0]=1.0
                else:
                    if theta < CI_lower_basic: num_lower_failures_basic += 1
                    elif theta > CI_upper_basic: num_upper_failures_basic += 1
                
                    
            
            elif mode=='empirical':  #boostrap CIs
                conf_level = coverage
                alpha = 100-conf_level
                
                CI_upper = np.percentile(theta_tildas, 100-alpha/2.0)
                CI_lower = np.percentile(theta_tildas, alpha/2.0)
    
                if theta >= CI_lower and theta <= CI_upper:
                    trial_results[t,0]=1.0
                else:
                    if theta < CI_lower: num_lower_failures += 1
                    elif theta > CI_upper: num_upper_failures += 1
                    
                q1 = np.percentile(theta_tildas, alpha/2.0)
                q2 = np.percentile(theta_tildas, 100-alpha/2.0)
                
                diff = q2 - q1
                moe = diff/2
                
                R = np.array([CI_lower, CI_upper])
                Rs[t,:] = R

                CI_upper_db = np.percentile(theta_tildas_db, 100-alpha/2.0)
                CI_lower_db = np.percentile(theta_tildas_db, alpha/2.0)

                if theta >= CI_lower_db and theta <= CI_upper_db:
                    trial_results_db[t,0]=1.0
                else:
                    if theta < CI_lower_db: num_lower_failures_db += 1
                    elif theta > CI_upper_db: num_upper_failures_db += 1

                R_db = np.array([CI_lower_db, CI_upper_db])
                Rs_db[t,:] = R_db
                
      
                CI_upper_naive = np.percentile(theta_tildas_naive, 100-alpha/2.0)
                CI_lower_naive = np.percentile(theta_tildas_naive, alpha/2.0)
    
                if theta >= CI_lower_naive and theta <= CI_upper_naive:
                    trial_results_naive[t,0]=1.0
                else:
                    if theta < CI_lower_naive: num_lower_failures_naive += 1
                    elif theta > CI_upper_naive: num_upper_failures_naive += 1
                    
                CI_upper_basic = np.percentile(theta_tildas_basic, 100-alpha/2.0)
                CI_lower_basic = np.percentile(theta_tildas_basic, alpha/2.0)
    
                if theta >= CI_lower_basic and theta <= CI_upper_basic:
                    trial_results_basic[t,0]=1.0
                else:
                    if theta < CI_lower_basic: num_lower_failures_basic += 1
                    elif theta > CI_upper_basic: num_upper_failures_basic += 1

            elif mode == 'studentized':  # studentized boostrap CIs
                conf_level = coverage
                alpha = 100 - conf_level

                studentized_vec = np.divide((theta_tildas - theta_priv), fishInfInvSqrtTildeVec)

                xi_upper = np.percentile(studentized_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(studentized_vec, alpha / 2.0)

                try:
                    CI_upper = (theta_priv - xi_lower * fishInfInvSqrt)[0]
                    CI_lower = (theta_priv - xi_upper * fishInfInvSqrt)[0]
                except:
                    CI_upper = (theta_priv - xi_lower * fishInfInvSqrt)
                    CI_lower = (theta_priv - xi_upper * fishInfInvSqrt)

                if theta >= CI_lower and theta <= CI_upper:
                    trial_results[t, 0] = 1.0
                else:
                    if theta < CI_lower:
                        num_lower_failures += 1
                    elif theta > CI_upper:
                        num_upper_failures += 1

                R = np.array([CI_lower, CI_upper])
                Rs[t, :] = R


                studentized_db_vec = np.divide((theta_tildas_db - theta_priv), fishInfInvSqrtTildeVec)

                xi_upper = np.percentile(studentized_db_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(studentized_db_vec, alpha / 2.0)

                CI_upper_db = theta_priv - xi_lower * fishInfInvSqrt
                CI_lower_db = theta_priv - xi_upper * fishInfInvSqrt

                R_db = np.array([CI_lower_db, CI_upper_db])
                Rs_db[t,:] = R_db

                if theta >= CI_lower_db and theta <= CI_upper_db:
                    trial_results_db[t, 0] = 1.0
                else:
                    if theta < CI_lower_db:
                        num_lower_failures_db += 1
                    elif theta > CI_upper_db:
                        num_upper_failures_db += 1

                studentized_naive_vec = np.divide((theta_tildas_naive - theta_priv), fishInfInvSqrtTildeVec)

                xi_upper = np.percentile(studentized_naive_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(studentized_naive_vec, alpha / 2.0)

                CI_upper_naive = theta_priv - xi_lower * fishInfInvSqrt
                CI_lower_naive = theta_priv - xi_upper * fishInfInvSqrt

                if theta >= CI_lower_naive and theta <= CI_upper_naive:
                    trial_results_naive[t, 0] = 1.0
                else:
                    if theta < CI_lower_naive:
                        num_lower_failures_naive += 1
                    elif theta > CI_upper_naive:
                        num_upper_failures_naive += 1

                studentized_basic_vec = np.divide(theta_tildas_basic - theta_priv, fishInfInvSqrtTildeVec)

                xi_upper = np.percentile(studentized_basic_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(studentized_basic_vec, alpha / 2.0)

                CI_upper_basic = theta_priv - xi_lower * fishInfInvSqrt
                CI_lower_basic = theta_priv - xi_upper * fishInfInvSqrt

                if theta >= CI_lower_basic and theta <= CI_upper_basic:
                    trial_results_basic[t, 0] = 1.0
                else:
                    if theta < CI_lower_basic:
                        num_lower_failures_basic += 1
                    elif theta > CI_upper_basic:
                        num_upper_failures_basic += 1

            elif mode == 'pivotal':  # studentized boostrap CIs

                conf_level = coverage
                alpha = 100 - conf_level

                pivotal_vec = theta_tildas - theta_priv

                xi_upper = np.percentile(pivotal_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(pivotal_vec, alpha / 2.0)

                # scaling factor should be 1
                try:
                    CI_upper = (theta_priv - xi_lower)[0]
                    CI_lower = (theta_priv - xi_upper)[0]
                except:
                    CI_upper = (theta_priv - xi_lower)
                    CI_lower = (theta_priv - xi_upper)


                if theta >= CI_lower and theta <= CI_upper:
                    trial_results[t, 0] = 1.0
                else:
                    if theta < CI_lower:
                        num_lower_failures += 1
                    elif theta > CI_upper:
                        num_upper_failures += 1

                R = np.array([CI_lower, CI_upper])
                Rs[t, :] = R

                pivotal_db_vec = theta_tildas_db - theta_priv

                xi_upper = np.percentile(pivotal_db_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(pivotal_db_vec, alpha / 2.0)

                CI_upper_db = theta_priv - xi_lower
                CI_lower_db = theta_priv - xi_upper

                R_db = np.array([CI_lower_db, CI_upper_db])
                Rs_db[t,:] = R_db.T

                if theta >= CI_lower_db and theta <= CI_upper_db:
                    trial_results_db[t, 0] = 1.0
                else:
                    if theta < CI_lower_db:
                        num_lower_failures_db += 1
                    elif theta > CI_upper_db:
                        num_upper_failures_db += 1

                pivotal_naive_vec = theta_tildas_naive - theta_priv

                xi_upper = np.percentile(pivotal_naive_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(pivotal_naive_vec, alpha / 2.0)

                CI_upper_naive = theta_priv - xi_lower
                CI_lower_naive = theta_priv - xi_upper

                if theta >= CI_lower_naive and theta <= CI_upper_naive:
                    trial_results_naive[t, 0] = 1.0
                else:
                    if theta < CI_lower_naive:
                        num_lower_failures_naive += 1
                    elif theta > CI_upper_naive:
                        num_upper_failures_naive += 1

                pivotal_basic_vec = theta_tildas_basic - theta_priv

                xi_upper = np.percentile(pivotal_basic_vec, 100 - alpha / 2.0)
                xi_lower = np.percentile(pivotal_basic_vec, alpha / 2.0)

                CI_upper_basic = theta_priv - xi_lower
                CI_lower_basic = theta_priv - xi_upper

                if theta >= CI_lower_basic and theta <= CI_upper_basic:
                    trial_results_basic[t, 0] = 1.0
                else:
                    if theta < CI_lower_basic:
                        num_lower_failures_basic += 1
                    elif theta > CI_upper_basic:
                        num_upper_failures_basic += 1

        # store results
        widths.append(np.mean(Rs, axis=0))
        widths_db.append(np.mean(Rs_db, axis=0))
        widths_FI.append(np.mean(Rs_FI, axis=0))
        widths_FI_np.append(np.mean(Rs_FI_np, axis=0))
        results.append(np.mean(trial_results, axis=0))
        results_db.append(np.mean(trial_results_db, axis=0))
        results_naive.append(np.mean(trial_results_naive, axis=0))
        results_basic.append(np.mean(trial_results_basic, axis=0))
        results_fisher.append(np.mean(trial_results_fisher, axis=0))
        results_fisher_np.append(np.mean(trial_results_fisher_np, axis=0))
        results_fisher_corr.append(np.mean(trial_results_fisher_corr, axis=0))
        
        list_upper_failures.append(num_upper_failures)
        list_lower_failures.append(num_lower_failures)
        list_upper_failures_db.append(num_upper_failures_db)
        list_lower_failures_db.append(num_lower_failures_db)
        list_upper_failures_naive.append(num_upper_failures_naive)
        list_lower_failures_naive.append(num_lower_failures_naive)
        list_upper_failures_basic.append(num_upper_failures_basic)
        list_lower_failures_basic.append(num_lower_failures_basic)
        list_upper_failures_fisher.append(num_upper_failures_fisher)
        list_lower_failures_fisher.append(num_lower_failures_fisher)
        list_upper_failures_fisher_np.append(num_upper_failures_fisher_np)
        list_lower_failures_fisher_np.append(num_lower_failures_fisher_np)
        list_upper_failures_fisher_corr.append(num_upper_failures_fisher_corr)
        list_lower_failures_fisher_corr.append(num_lower_failures_fisher_corr)


    print("Private Parametric Bootstrap:")
    print([a[0] for a in results])
    print("Private Parametric Bootstrap De-biased:")
    print([a[0] for a in results_db])
    # print("\n")
    # # print("Non-Private Parametric Bootstrap:")
    # # #print(results_naive)
    # # print([a[0] for a in results_basic])
    # print("\n")
    print("Fisher Information with theta private:")
    print([a[0] for a in results_fisher])
    print("\n")
    print("Fisher Information non-private:")
    print([a[0] for a in results_fisher_np])
    print("\n")
    # print("Fisher Information corrected:")
    # print([a[0] for a in results_fisher_corr])
    # print("\n")
    
    #save results

    output = {}

    name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '_clampT_' + str(cliphi)

    output['results'] = results
    output['upperfailures'] = list_upper_failures
    output['lowerfailures'] = list_lower_failures
    output['widths'] = widths

    output['results_DEBIASED'] = results_db
    output['upperfailures_DEBIASED'] = list_upper_failures_db
    output['lowerfailures_DEBIASED'] = list_lower_failures_db
    output['widths_DEBIASED'] = widths_db

    output['results_NAIVE'] = results_naive
    output['upperfailures_NAIVE'] = list_upper_failures_naive
    output['lowerfailures_NAIVE'] = list_lower_failures_naive

    output['results_BASIC'] = results_basic
    output['upperfailures_BASIC'] = list_upper_failures_basic
    output['lowerfailures_BASIC'] = list_lower_failures_basic

    output['results_FISHER'] = results_fisher
    output['upperfailures_FISHER'] = list_upper_failures_fisher
    output['lowerfailures_FISHER'] = list_lower_failures_fisher
    output['widths_FISHER'] = widths_FI

    output['results_FISHERNP'] = results_fisher_np
    output['upperfailures_FISHERNP'] = list_upper_failures_fisher_np
    output['lowerfailures_FISHERNP'] = list_lower_failures_fisher_np
    output['widths_FISHERNP'] = widths_FI_np
    
    #name_suffix = distribution + '_' + 'N'+str(N) + '_' + 'epsilon'+str(noise_scale) + '_' + mode + '_clampT_' + str(cliphi) +  '_FISHERCORR.npy'
    #np.save('results_' + name_suffix, results_fisher_corr)
    #np.save('upperfailures_' + name_suffix, list_upper_failures_fisher_corr)
    #np.save('lowerfailures_' + name_suffix, list_lower_failures_fisher_corr)

    with open(name_suffix + '.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(name_suffix + '.pickle', 'rb') as handle:
        b = pickle.load(handle)
        print(b)

    
if __name__ == "__main__":
    
    np.random.seed(22)
    
    parser = argparse.ArgumentParser(description='Confidence Intervals for Private Estimators')
    
    parser.add_argument('--N', type=int, default=50, help='data size')
    parser.add_argument('--d', type=str, default='gamma', help='distribution (poisson, gaussian, gamma, gaussian2)')
    parser.add_argument('--mode', type=str, default='pivotal', help='analytic or empirical (CI mode)')
    parser.add_argument('--e', type=float, default=0.5, help='DP epsilon')
    parser.add_argument('--cliplo', type=float, default=0, help='clip lowerbound')
    parser.add_argument('--cliphi', type=float, default=np.inf, help='clip upperbound')
    parser.add_argument('--rho', type=float, default=0.85, help='privacy budget split if more than one parameter to privatize')
    parser.print_help()
    
    args = parser.parse_args()
    
    theta = np.random.rand() * 20
    theta2 = np.random.rand() * 8


    CIs(args.d, [theta, theta2], args.N, 1000, args.e, args.mode, 1000, args.cliplo, args.cliphi, args.rho)
