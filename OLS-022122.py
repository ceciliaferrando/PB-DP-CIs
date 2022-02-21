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
    assert (np.all(ll > 0))

    return xPSD

def generate_ols_data(N, D, true_beta, gamma, delta, zeta):
    '''
    :param N: number of data points
    :param D: number of dimension
    :param true_beta: true OLS arameter
    :param gamma: bound on x, the ols data
    :param delta: bound on u, the ols noise
    :return: X, U, Y (generated dataset ), beta (generated coefficient beta)
    '''
    boundY = [-zeta, zeta]
    beta = np.array([true_beta]*D)
    X = np.random.uniform(-gamma, gamma, (N, D))
    # uniformly generate N error terms
    U = np.random.uniform(-delta, delta, N)
    # step four construct the Y's
    Y = np.dot(X, beta) + U
    #disard data points that violate the bound on Y
    indices = np.hstack((np.where(Y<boundY[0]), np.where(Y>boundY[1])))
    X = np.delete(X, indices, axis = 0)
    Y = np.delete(Y, indices)

    return X, U, Y, beta


def compute_dp_noise(D, gamma, zeta, eps):
    '''
    :param gs_XY: global sensitivity computed on XtY
    :param gs_XX: global sensitivity computed on XtX
    :param D: dimension of ols
    :return: w, the laplace noise to be added on XY term, a D-dim vector
             v, the laplace noise to be added on XX term, a D-D matrix
    '''
    Delta_w = D * 2*gamma * 2*zeta
    w = np.zeros((D,))
    for i_w in range(len(w)):
        w[i_w] = np.random.laplace(0, Delta_w / eps/3, 1)

    Delta_V = D*(D + 1) * gamma**2
    V = np.zeros((D, D))
    for i_x in range(D):
        for j_x in range(D):
            if i_x <= j_x:
                V[i_x][j_x] = np.random.laplace(0, Delta_V / eps/3, 1)
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

def private_OLS(X, y, args):
    """
    :param X: independent variable data
    :param y: dependent variable data
    :param U: errors
    :param beta_true: true beta coefficient
    :param args: arguments
    :return: private estimates of beta, Q, sigma; sensitivity of XtX, sensitivity of XtY
    """

    w, V = compute_dp_noise(args.D, args.gamma, args.zeta, args.eps)
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)

    beta_hat = np.dot(inv(XtX), Xty)  # this is the non-private estimate for beta
    beta_hat_priv = np.dot(inv(XtX + V), (Xty + w))

    Q_hat = 1 / args.N * XtX
    Q_hat_priv = Q_hat + 1 / args.N * V
    if is_pos_def(Q_hat_priv) == False:
        Q_hat_priv = make_pos_def(Q_hat_priv, small_positive=0.1)

    upper_bound_x, lower_bound_x = args.gamma, -args.gamma
    upper_bound_y, lower_bound_y = args.zeta, -args.zeta
    width_term = max((upper_bound_y - np.sum(lower_bound_x * np.abs(beta_hat))) ** 2,
                     (lower_bound_y - np.sum(upper_bound_x * np.abs(beta_hat))) ** 2)
    Delta_sigma_sq = 1.0 / (args.N - args.D) * width_term
    sigma_sq_hat_priv = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat)) ** 2) + \
                        np.random.laplace(0, Delta_sigma_sq / args.eps / 3)  ### note ###

    if sigma_sq_hat_priv < 0:
        sigma_sq_hat_priv = 0.1

    return (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv)


def hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, args):
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

        w_star = compute_dp_noise(args.D, args.gamma, args.zeta, args.eps)[0]
        V_star = compute_dp_noise(args.D, args.gamma, args.zeta, args.eps)[1]

        cov_matrix = sigma_sq_hat_priv * Q_hat_priv
        # if is_pos_def(cov_matrix) == False:
        #     cov_matrix = make_pos_def(cov_matrix)
        Z_star = np.random.multivariate_normal(np.zeros(args.D), cov_matrix)

        Q_hat_star = Q_hat_priv + 1 / args.N * V_star

        beta_star_b = np.dot(np.dot(inv(Q_hat_star), Q_hat_priv), beta_hat_priv) + \
                      np.dot(inv(Q_hat_star), 1 / np.sqrt(args.N) * Z_star + 1 / args.N * w_star)
        beta_star_vec[b, :] = beta_star_b

    return beta_star_vec


def coverage_test_single_trial(args):
    """
    :param args: input arguments
    :return: dictionary with results from a single trial
    """

    X, U, y, beta_true_vec = generate_ols_data(args.N, args.D, args.true_beta, args.gamma, args.delta, args.zeta)

    dict_result = defaultdict(dict)

    # standard (public) method

    XtY = np.dot(X.T, y)
    XtX = np.dot(X.T, X)
    beta_hat = np.dot(inv(XtX), XtY)  # this is the non-private estimate for beta
    sigma_sq_est = 1.0 / (args.N - args.D) * np.sum((y - np.dot(X, beta_hat)) ** 2)  # non-private estimate for sigma
    dist_std = np.sqrt(1.0 * sigma_sq_est / XtX[args.test_d][args.test_d])  # STD in the distribution used to get CI

    CI_lower = beta_hat[args.test_d] - args.z_score * dist_std
    CI_upper = beta_hat[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['standard'] = determine_trial_result(CI_lower, CI_upper, beta_true_vec[args.test_d])
    dict_result['ci']['standard'] = (CI_lower, CI_upper)
    dict_result['estimate']['standard'] = beta_hat

    # naive private method (Fisher CIs)

    (beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv) = private_OLS(X, y, args)

    dist_std = np.sqrt(1.0 * sigma_sq_hat_priv / (args.N * Q_hat_priv[args.test_d][args.test_d]))

    CI_lower_priv = beta_hat_priv[args.test_d] - args.z_score * dist_std
    CI_upper_priv = beta_hat_priv[args.test_d] + args.z_score * dist_std
    dict_result['coverage']['private'] = determine_trial_result(CI_lower_priv, CI_upper_priv, beta_true_vec[args.test_d])
    dict_result['ci']['private'] = (CI_lower_priv, CI_upper_priv)
    dict_result['estimate']['private'] = beta_hat_priv

    # hybrid bootstrap private method

    bootstrap_vec = hybrid_bootstrap_OLS(beta_hat_priv, Q_hat_priv, sigma_sq_hat_priv, args)

    empirical_CIs = np.zeros((args.D, 2))
    for i in range(args.D):
        empirical_CIs[i][0] = np.percentile(bootstrap_vec[:, i], (100 - args.coverage) / 2)
        empirical_CIs[i][1] = np.percentile(bootstrap_vec[:, i], 100 - (100 - args.coverage) / 2)

    CI_lower_bootstrap = empirical_CIs[args.test_d][0]
    CI_upper_bootstrap = empirical_CIs[args.test_d][1]

    dict_result['coverage']['bootstrap'] = determine_trial_result(CI_lower_bootstrap, CI_upper_bootstrap,
                                                                  beta_true_vec[args.test_d])
    dict_result['ci']['bootstrap'] = (CI_lower_bootstrap, CI_upper_bootstrap)
    dict_result['estimate']['bootstrap'] = np.mean(bootstrap_vec, axis=0)

    return dict_result


######################################################################
# DRIVER
######################################################################
np.random.seed(21)
parser = argparse.ArgumentParser()
parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of experiments to compute quantile')
parser.add_argument('--eps', type=float, default=5, help='privacy threshold args.epsilon')
parser.add_argument('--gamma', type=int, default=5, help='bound on x')
parser.add_argument('--delta', type=float, default=10, help='bound on u')
parser.add_argument('--zeta', type=float, default=150, help='bound on Y')
parser.add_argument('--true_beta', type=float, default=10.0, help='true beta')
parser.add_argument('--z_score', type=float, default=1.96,
                    help='z score associated with confidence level')  # 1.96 for 95% confidence level
parser.add_argument('--num_trials', type=int, default=1000, help='Number of experiments to compute coverage ratio')
parser.add_argument('--coverage', type=int, default=95, help='The dimension to be looked at')
parser.add_argument('--test_d', type=int, default=2, help='The dimension to be looked at')
parser.add_argument('--D', type=int, default=4, help='The number of dimensions in the ols problem')
parser.add_argument('--N', type=int, default=100, help='The number of data points in the ols problem')

args = parser.parse_args()
print(args)

if __name__ == "__main__":

    data_dir = 'data022122c/'

    save_dir = data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    list_ci_levels = sorted(list(z_values.keys()))
    args.z_score = z_values[args.coverage]

    methods = ['standard', 'private', 'bootstrap']

    print(args)
    N_list = [10, 50, 100, 500, 1000, 5000, 10000]
    for N_idx, N in enumerate(N_list):

        trial_results = np.zeros((args.num_trials,2))
        trial_results_naive = np.zeros((args.num_trials,2))
        trial_results_standard = np.zeros((args.num_trials,2))

        trial_widths = np.zeros((args.num_trials, 2))
        trial_widths_naive = np.zeros((args.num_trials, 2))
        trial_widths_standard = np.zeros((args.num_trials, 2))


        #print('\n\n')
        #print(f'Test case {N_idx + 1}/{ len(N_list) } when N = {N}')
        exp_dict = {}
        result = {}

        args.N = N

        inspection_interval = np.ceil(args.num_trials / 20)
        for trial_idx in range(args.num_trials):
            # standard
            out = coverage_test_single_trial(args)

            trial_results[trial_idx, 0] = 1 if out['coverage']['bootstrap'] == 0 else 0
            trial_results_naive[trial_idx,0] = 1 if out['coverage']['private'] == 0 else 0
            trial_results_standard[trial_idx, 0] = 1 if out['coverage']['standard'] == 0 else 0

            trial_results[trial_idx, 1] = out['estimate']['bootstrap'][args.test_d]
            trial_results_naive[trial_idx,1] = out['estimate']['private'][args.test_d]
            trial_results_standard[trial_idx,1] = out['estimate']['standard'][args.test_d]

            trial_widths[trial_idx,:] = out['ci']['bootstrap']
            trial_widths_naive[trial_idx,:] = out['ci']['private']
            trial_widths_standard[trial_idx,:] = out['ci']['standard']

            if trial_idx % inspection_interval == 0:
                print(f'\t\t {trial_idx} / {args.num_trials} done')

        output = {}
        name_suffix = 'OLS' + '_' + 'N' + str(N) + '_' + 'epsilon' + str(args.eps)
        output['results'] = np.mean(trial_results,axis=0)
        output['widths'] = np.mean(trial_widths, axis=0)
        output['results_NAIVE'] = np.mean(trial_results_naive,axis=0)
        output['widths_NAIVE'] = np.mean(trial_widths_naive, axis=0)
        output['results_FISHERNP'] = np.mean(trial_results_standard,axis=0)
        output['widths_FISHERNP'] = np.mean(trial_widths_standard, axis=0)

        with open(name_suffix + '.pickle', 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(name_suffix + '.pickle', 'rb') as handle:
            b = pickle.load(handle)
            print(b)

    print('Done')