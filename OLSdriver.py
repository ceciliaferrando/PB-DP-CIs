import argparse
import os, pickle
import numpy as np
from numpy.linalg import inv
from collections import defaultdict
from OLSfunctions import *

def coverage_test_single_trial(args):

    X, U, y, beta_true = generate_ols_data(args.N, args.D, args.bound_beta, args.gamma, args.delta)

    dict_result = defaultdict(dict)

    # standard
    XtY = np.dot(X.T, y)
    XtX = np.dot(X.T, X)
    beta_hat = np.dot(inv(XtX), XtY)  # this is the non-private estimate for beta
    sigma_sq_est = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat)) ** 2)  # non-private estimate for sigma
    dist_std = np.sqrt(1.0 * sigma_sq_est / XtX[args.test_d][args.test_d])  # STD in the distribution used to get CI

    CI_lower = beta_hat[args.test_d] - args.z_score * dist_std
    CI_upper = beta_hat[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['standard'] = determine_trial_result(CI_lower, CI_upper, beta_true[args.test_d])
    dict_result['ci']['standard'] = (CI_lower, CI_upper)

    (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY) = private_OLS(X, y, U, beta_true, args)

    # naive private

    upper_bound_y = np.max(y)
    lower_bound_x_beta = np.abs(np.sum(args.gamma * np.abs(beta_hat_priv)))
    gf_sigma_sq = 1.0 / (args.N - args.D) * (upper_bound_y + lower_bound_x_beta)  ### note ###
    sigma_sq_est_priv = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat_priv)) ** 2) + np.random.laplace(0, gf_sigma_sq / args.eps)  ### note ###

    dist_std = np.sqrt(1.0 * sigma_sq_est_priv / XtX[args.test_d][args.test_d])  # STD in the distribution used to get CI cecilia -> this uses XtX!

    CI_lower_priv = beta_hat_priv[args.test_d] - args.z_score * dist_std
    CI_upper_priv = beta_hat_priv[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['private'] = determine_trial_result(CI_lower_priv, CI_upper_priv, beta_true[args.test_d])
    dict_result['ci']['private'] = (CI_lower_priv, CI_upper_priv)


    # bootstrap private

    bootstrap_vec = hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY, args)

    empirical_CIs = np.zeros((args.D, 2))
    for i in range(args.D):
        empirical_CIs[i][0] = np.percentile(bootstrap_vec[:, i], (100 - args.coverage)/2)
        empirical_CIs[i][1] = np.percentile(bootstrap_vec[:, i], 100 - (100 - args.coverage)/2)

    CI_lower_bootstrap = empirical_CIs[args.test_d][0]
    CI_upper_bootstrap = empirical_CIs[args.test_d][1]

    dict_result['coverage']['bootstrap'] = determine_trial_result(CI_lower_bootstrap, CI_upper_bootstrap, beta_true[args.test_d])
    dict_result['ci']['bootstrap'] = (CI_lower_bootstrap, CI_upper_bootstrap)

    return dict_result

######################################################################
# DRIVER
######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of experiments to compute quantile')
parser.add_argument('--eps', type=float, default=3, help='privacy threshold args.epsilon')
parser.add_argument('--gamma', type=int, default=5, help='bound on x')
parser.add_argument('--delta', type=float, default=10, help='bound on u')
parser.add_argument('--bound_beta', type=float, default=10.0, help='bound on beta')
parser.add_argument('--z_score', type=float, default=1.96, help='z score associated with confidence level') # 1.96 for 95% confidence level
parser.add_argument('--num_trials', type=int, default=1000, help='Number of experiments to compute coverage ratio')
parser.add_argument('--coverage', type=int, default=99, help='The dimension to be looked at')

parser.add_argument('--test_d', type=int, default=2, help='The dimension to be looked at')
parser.add_argument('--D', type=int, default=4, help='The number of dimensions in the ols problem')
parser.add_argument('--N', type=int, default=100, help='The number of data points in the ols problem')

args = parser.parse_args()
print(args)

# I haven't changed the main call too much

if __name__ == "__main__":

    data_dir = './data/'
    plot_dir = './plot/'

    save_dir = data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    list_ci_levels = sorted(list(z_values.keys()))
    args.z_score = z_values[args.coverage]

    methods = ['standard', 'private', 'bootstrap']

    print(args)
    N_list = [50, 100, 500, 1000, 5000, 10000]
    for N_idx, N in enumerate(N_list):
        #print('\n\n')
        #print(f'Test case {N_idx + 1}/{ len(N_list) } when N = {N}')
        exp_dict = {}
        result = {}

        args.N = N
        trial_result = []

        inspection_interval = np.ceil(args.num_trials / 20)
        for trial_idx in range(args.num_trials):
            # standard
            trial_result.append(coverage_test_single_trial(args))

            if trial_idx % inspection_interval == 0:
                print(f'\t\t {trial_idx} / {args.num_trials} done')

        print(trial_result)

        coverage_result = {}
        ci_result = {}
        for method in methods:
            coverage_result[method] = [item['coverage'][method] for item in trial_result] # group the counts over all trials
            # counter_info = Counter(coverage_result[method])

            ci_result[method] = [item['ci'][method] for item in trial_result]

        result['coverage'] = coverage_result
        result['ci'] = ci_result

        exp_dict = {'args': args, 'result': result}
        with open( os.path.join(save_dir, f'result_cov_{args.coverage}_N_{args.N}.pkl'), 'wb' ) as f:
            pickle.dump(exp_dict, f)
            print('\t\t' + f'result_cov_{args.coverage}_N_{args.N}.pkl' + f' saved in {save_dir}')

    print('Done')