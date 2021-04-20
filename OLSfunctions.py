import argparse
import os, pickle
import numpy as np
from numpy.linalg import inv
from collections import defaultdict


#########################################################################################################
# HELPERS
#########################################################################################################

# generate the ols data and ground truth coeff
def generate_ols_data(N, D, bound_beta, gamma, delta):
    '''
    :param N: number of data points
    :param D: number of dimension
    :param bound_beta: bound on beta, the ols coefficients
    :param gamma: bound on x, the ols data
    :param delta: bound on u, the ols noise
    :return: X, U, Y (generated dataset ), beta (generated coefficient beta)
    '''
    beta = np.random.uniform(bound_beta, bound_beta, D) # true beta
    X = np.random.uniform(-gamma, gamma, (N, D))
    # uniformly generate N error terms
    U = np.random.uniform(-delta, delta, N)
    # step four construct the Y's
    Y = np.dot(X, beta) + U
    return X, U, Y, beta

def compute_sensitivity(beta, gamma, D, delta):
    '''
    :param beta: the ols coefficients
    :param gamma: bound on x, the ols data
    :param D: number of dimension
    :param delta: bound on u, the ols noise
    :return: Delta_XX, the global sensitivty on XtX; Delta_XY, the global sensitivity on XtY
    '''
    Delta_XY = beta * gamma * gamma + 2 * gamma * delta
    Delta_XX = np.ones((D, D)) * (gamma ** 2) * 2
    for i_x in range(D):
        Delta_XX[i_x][i_x] = gamma ** 2
    return Delta_XX, Delta_XY

def compute_dp_noise(Delta_XY, Delta_XX, D, eps):
    '''
    :param gs_XY: global sensitivity computed on XtY
    :param gs_XX: global sensitivity computed on XtX
    :param D: dimension of ols
    :return: w, the laplace noise to be added on XY term, a D-dim vector
             v, the laplace noise to be added on XX term, a D-D matrix
    '''
    w = np.zeros_like(Delta_XY)
    for i_w in range(len(w)):
        w[i_w] = np.random.laplace(0, Delta_XY[i_w] / eps, 1)

    v = np.zeros((D, D))
    for i_x in range(D):
        for j_x in range(D):
            if i_x <= j_x:
                v[i_x][j_x] = np.random.laplace(0, Delta_XX[i_x][j_x] / eps, 1)
                v[j_x][i_x] = v[i_x][j_x]

    return w, v

def determine_trial_result(CI_lower, CI_upper, estimation):
    '''
    :param CI_lower:
    :param CI_upper:
    :param estimation: -1 forlowererror, 1 for upper error, 0 for correct coverage
    :return:
    '''
    if estimation < CI_lower:
        return -1
    elif estimation > CI_upper:
        return 1
    else:
        return 0


#########################################################################################################
# CORE FUNCTIONS
#########################################################################################################


def private_OLS(X, y, U, beta_true, args):

    N = args.N

    # compute sensitivity
    Delta_XX, Delta_XY = compute_sensitivity(beta_true, args.gamma, args.D, args.delta)
    # cecilia -> is it find to use beta_true to compute sensitivity?
    # sample calibrated privacy noise
    w, V = compute_dp_noise(Delta_XY, Delta_XX, args.D, args.eps)
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)

    # obtain privatized beta_hat_priv
    beta_hat_priv = np.dot(inv(XtX + V), (Xty + w))  # this is the private estimate for beta

    # obtain privatized Q_hat_priv
    Q_hat = 1/args.N * XtX
    Q_hat_priv = Q_hat + 1/args.N * V

    joint_ex_sq = (1.0 / N) * np.dot((X ** 2).T, (X ** 2)) + np.random.laplace(0, (1.0 / N) * (
                args.gamma * args.gamma * args.gamma * args.gamma / args.eps), (args.D, args.D))

    # obtain privatized sigma_sq_hat
    upper_bound_y = np.max(y)
    lower_bound_x_beta = np.abs(np.sum(args.gamma * np.abs(beta_hat_priv)))
    gf_sigma_sq = 1.0 / (args.N - args.D) * (upper_bound_y + lower_bound_x_beta)  ### note ###
    noise_sigma = np.random.laplace(0, gf_sigma_sq / args.eps)
    sigma_sq_hat_priv = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat_priv)) ** 2) + noise_sigma  ### note ###

    return(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY)

def hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY, args):

    D = args.D
    N = args.N
    eps = args.eps

    beta_tilde_vec = np.zeros((args.num_bootstraps, D))

    for b in range(args.num_bootstraps):

        # SIMULATE INGREDIENTS

        # simulate N_inv_V_tilde
        N_inv_v_tilde = np.zeros((D, D))
        for i_v in range(D):
            for j_v in range(D):
                if i_v <= j_v:
                    N_inv_v_tilde[i_v][j_v] = np.random.laplace(0, Delta_XX[i_v][j_v] / (N * eps))
                    N_inv_v_tilde[j_v][i_v] = N_inv_v_tilde[i_v][j_v]

        # simulate Z
        sqrtN_inv_Z = 1/np.sqrt(N) * np.random.multivariate_normal(np.zeros(D), sigma_sq_hat_priv * Q_hat_priv)   # cecilia -> check this w/ Dan

        # simulate N_inv_w
        N_inv_w_tilde = np.zeros(D)
        for i_w in range(len(N_inv_w_tilde)):
            N_inv_w_tilde[i_w] = np.random.laplace(0, (1.0 / N) * (Delta_XY[i_w] / eps), 1)

        # CONSTRUCT THE TERMS

        term_1 = Q_hat_priv + N_inv_v_tilde
        term_2 = sqrtN_inv_Z + N_inv_w_tilde

        # OBTAIN beta_tilde_b

        beta_tilde_b = np.dot(np.dot(inv(term_1), Q_hat_priv), beta_hat_priv) + np.dot(inv(term_1), term_2)
        beta_tilde_vec[b, :] = beta_tilde_b

    return beta_tilde_vec

