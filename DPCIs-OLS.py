import argparse
import os, pickle
import numpy as np
from numpy.linalg import inv
from collections import defaultdict

############################################################################
# HELPERS
############################################################################

def is_pos_def(x):
    """
    :param x: input matrix
    :return: True if x is PSD
    """
    return np.all(np.linalg.eigvals(x) > 0)

def make_pos_def(x, small_positive=0.1):
    """
    :param x: input matrix
    :param small_positive: float
    :return: PSD projection of x
    """

    # Compute SVD and eigenvalue decompositions
    (u, s, v) = np.linalg.svd(x)
    (l, w) = np.linalg.eig(x)

    # Make sure x is not PSD
    if np.all(l >= 0):
        raise ValueError("X is already PSD")
    l_prime = np.where(l > 0, l, small_positive)
    xPSD = w.dot(np.diag(l_prime)).dot(w.T)

    # Check
    ll, _ = np.linalg.eig(xPSD)
    assert(np.all(ll > 0))

    return xPSD


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
    U = np.random.uniform(-delta, delta, N)
    Y = np.dot(X, beta) + U

    return X, U, Y, beta

def compute_sensitivity(beta, gamma, D, delta):
    '''
    :param beta: the ols coefficients
    :param gamma: bound on x, the ols data
    :param D: number of dimension
    :param delta: bound on u, the ols noise
    :return: Delta_XX, the global sensitivity on XtX; Delta_XY, the global sensitivity on XtY
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

    V = np.zeros((D, D))
    for i_x in range(D):
        for j_x in range(D):
            if i_x <= j_x:
                V[i_x][j_x] = np.random.laplace(0, Delta_XX[i_x][j_x] / eps, 1)
                V[j_x][i_x] = V[i_x][j_x]

    return w, V


def determine_trial_result(CI_lower, CI_upper, value):
    """
    :param CI_lower: confidence interval lower bound
    :param CI_upper: confidence interval upper bound
    :param value: value to check (does the confidence interval include the value?)
    :return: 1 if the CI covers the value, 0 otherwise
    """

    if value < CI_lower:
        return -1
    elif value > CI_upper:
        return 1
    else:
        return 0
    
############################################################################
# CORE FUNCTIONS
############################################################################

def private_OLS(X, y, U, beta_true, args):
    """
    :param X: independent variable data
    :param y: dependent variable data
    :param U: errors
    :param beta_true: true beta coefficient
    :param args: arguments
    :return: private estimates of beta, Q, sigma; sensitivity of XtX, sensitivity of XtY
    """

    Delta_XX, Delta_XY = compute_sensitivity(beta_true, args.gamma, args.D, args.delta)
    w, V = compute_dp_noise(Delta_XY, Delta_XX, args.D, args.eps/3)
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)

    beta_hat_priv = np.dot(inv(XtX + V), (Xty + w))

    Q_hat = 1/args.N * XtX
    Q_hat_priv = Q_hat + 1/args.N * V

    upper_bound_y = np.max(y)
    lower_bound_x_beta = np.abs(np.sum(args.gamma * np.abs(beta_hat_priv)))
    gf_sigma_sq = 1.0 / (args.N - args.D) * (upper_bound_y + lower_bound_x_beta)
    noise_sigma = np.random.laplace(0, gf_sigma_sq / args.eps/3)
    sigma_sq_hat_priv = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat_priv)) ** 2) + noise_sigma

    return(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY)


def hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY, args):
    """
    :param beta_hat_priv: private estimate of beta
    :param Q_hat_priv: private estimate of Q
    :param sigma_sq_hat_priv: private estimate of sigma**2
    :param Delta_XX: global sensitivity of XtX
    :param Delta_XY: global sesitivity of XtY
    :param args: input arguments
    :return: vector of bootstrap beta estimates
    """

    beta_star_vec = np.zeros((args.num_bootstraps, args.D))

    for b in range(args.num_bootstraps):

        w_star = compute_dp_noise(Delta_XY, Delta_XX, args.D, args.eps/3)[0]
        V_star = compute_dp_noise(Delta_XY, Delta_XX, args.D, args.eps/3)[1]


        cov_matrix = sigma_sq_hat_priv * Q_hat_priv
        if is_pos_def(cov_matrix) == False:
            cov_matrix = make_pos_def(cov_matrix)
        Z_star = np.random.multivariate_normal(np.zeros(args.D), cov_matrix)

        Q_hat_star = Q_hat_priv + 1/args.N * V_star

        beta_star_b = np.dot(np.dot(inv(Q_hat_star), Q_hat_priv), beta_hat_priv) + \
                       np.dot(inv(Q_hat_star), 1/np.sqrt(args.N) * Z_star + 1/args.N * w_star)
        beta_star_vec[b, :] = beta_star_b

    return beta_star_vec


def coverage_test_single_trial(args):
    """
    :param args: input arguments
    :return: dictionary with results from a single trial
    """

    X, U, y, beta_true = generate_ols_data(args.N, args.D, args.bound_beta, args.gamma, args.delta)

    dict_result = defaultdict(dict)

    # standard (public) method

    XtY = np.dot(X.T, y)
    XtX = np.dot(X.T, X)
    beta_hat = np.dot(inv(XtX), XtY)  # this is the non-private estimate for beta
    sigma_sq_est = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat)) ** 2)  # non-private estimate for sigma
    dist_std = np.sqrt(1.0 * sigma_sq_est / XtX[args.test_d][args.test_d])  # STD in the distribution used to get CI

    CI_lower = beta_hat[args.test_d] - args.z_score * dist_std
    CI_upper = beta_hat[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['standard'] = determine_trial_result(CI_lower, CI_upper, beta_true[args.test_d])
    dict_result['ci']['standard'] = (CI_lower, CI_upper)

    # naive private method (Fisher CIs)

    (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, Delta_XX, Delta_XY) = private_OLS(X, y, U, beta_true, args)

    upper_bound_y = np.max(y)
    lower_bound_x_beta = np.abs(np.sum(args.gamma * np.abs(beta_hat_priv)))
    dist_std = np.sqrt(1.0 * sigma_sq_hat_priv / (args.N*Q_hat_priv[args.test_d][args.test_d]))

    CI_lower_priv = beta_hat_priv[args.test_d] - args.z_score * dist_std
    CI_upper_priv = beta_hat_priv[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['private'] = determine_trial_result(CI_lower_priv, CI_upper_priv, beta_true[args.test_d])
    dict_result['ci']['private'] = (CI_lower_priv, CI_upper_priv)

    # hybrid bootstrap private method

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
parser.add_argument('--coverage', type=int, default=50, help='The dimension to be looked at')
parser.add_argument('--test_d', type=int, default=2, help='The dimension to be looked at')
parser.add_argument('--D', type=int, default=4, help='The number of dimensions in the ols problem')
parser.add_argument('--N', type=int, default=100, help='The number of data points in the ols problem')

args = parser.parse_args()
print(args)

if __name__ == "__main__":

    data_dir = './data022022/'
    plot_dir = './plot/'

    save_dir = data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    list_ci_levels = sorted(list(z_values.keys()))
    args.z_score = z_values[args.coverage]

    methods = ['standard', 'private', 'bootstrap']

    print("arguments", args)
    N_list = [50, 100, 500, 1000, 5000, 10000]
    for N_idx, N in enumerate(N_list):
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
            coverage_result[method] = [item['coverage'][method] for item in trial_result] 
            ci_result[method] = [item['ci'][method] for item in trial_result]

        result['coverage'] = coverage_result
        result['ci'] = ci_result

        exp_dict = {'args': args, 'result': result}
        with open( os.path.join(save_dir, f'result_cov_{args.coverage}_N_{args.N}.pkl'), 'wb' ) as f:
            pickle.dump(exp_dict, f)
            print('\t\t' + f'result_cov_{args.coverage}_N_{args.N}.pkl' + f' saved in {save_dir}')

    print('Done')
