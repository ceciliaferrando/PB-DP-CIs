import argparse
import os, pickle
import numpy as np
from numpy.linalg import inv
from collections import defaultdict

############################################################################
# HELPERS
############################################################################

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def make_pos_def(x, small_positive=0.1):

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

def bootstrap_procedure(beta_hat_priv, gs_XY, gs_XX, sigma_sq_est, ex, ex_sq, ex_4, joint_ex, joint_ex_sq, args):
    D = args.D
    N = args.N
    eps = args.eps
    empirical_quantile = np.zeros((D, 2))
    empirical_quantile_danl = np.zeros((D, 2))

    beta_sim = np.zeros((args.num_bootstraps, D))
    beta_sim_dan_l = np.zeros((args.num_bootstraps, D))

    for n_rv in range(args.num_bootstraps):
        N_inv_XtX_sim = np.zeros((D, D))
        N_inv_v_sim = np.zeros((D,D))
        N_inv_v_tilde_sim = np.zeros((D, D))
        N_inv_w_sim = np.zeros(D)
        N_inv_XtU_sim = np.zeros(D)

        # simulate N_inv_XtX
        for i in range(D):
            for j in range(D):
                if i < j:
                    N_inv_XtX_sim[i][j] = np.random.normal(joint_ex[i][j],
                                        np.sqrt( np.abs(joint_ex_sq[i][j] - (joint_ex[i][j] ** 2) ) / N ), 1)
                    N_inv_XtX_sim[j][i] = N_inv_XtX_sim[i][j]
                if i == j:
                    N_inv_XtX_sim[i][j] = np.random.normal(ex_sq[i], np.sqrt( np.abs(ex_4[i] - ex_sq[i] ** 2) / N), 1)

        # simulate N_inv_V
        for i_v in range(D):
            for j_v in range(D):
                if i_v <= j_v:
                    N_inv_v_sim[i_v][j_v] = np.random.laplace(0, gs_XX[i_v][j_v] / (N * eps))
                    N_inv_v_sim[j_v][i_v] = N_inv_v_sim[i_v][j_v]

        # simulate N_inv_V_tilde
        for i_v in range(D):
            for j_v in range(D):
                if i_v <= j_v:
                    N_inv_v_tilde_sim[i_v][j_v] = np.random.laplace(0, gs_XX[i_v][j_v] / (N * eps))
                    N_inv_v_tilde_sim[j_v][i_v] = N_inv_v_tilde_sim[i_v][j_v]

        # simulate N_inv_w
        for i_w in range(len(N_inv_w_sim)):
            N_inv_w_sim[i_w] = np.random.laplace(0, (1.0 / N) * (gs_XY[i_w] / eps), 1)

        # simulate N_inv_XtU
        print(is_pos_def(sigma_sq_est * N_inv_XtX_sim * (1.0/N)))
        N_inv_XtU_sim = np.random.multivariate_normal(np.zeros(D), sigma_sq_est * N_inv_XtX_sim * (1.0/N) )
        # cecilia -> question: should sigma_sq_est be the private one

        B = np.dot(inv(N_inv_XtX_sim + N_inv_v_sim), (N_inv_XtU_sim + N_inv_w_sim) )
        A = np.dot(inv(N_inv_XtX_sim + N_inv_v_sim), N_inv_XtX_sim)

        # print(A)
        beta_sim[n_rv, :] = np.dot(inv(A), (beta_hat_priv - B))

        # CECILIA:
        # B = np.dot(inv(N_inv_XtX_sim + N_inv_v_sim + N_inv_v_tilde_sim), (N_inv_XtU_sim + N_inv_w_sim))
        # A = np.dot(inv(N_inv_XtX_sim + N_inv_v_sim + N_inv_v_tilde_sim), (N_inv_XtX_sim + N_inv_v_sim))
        # beta_sim[n_rv, :] = np.dot(inv(A), (beta_hat_priv - B))

        # cecilia: comment - > I changed the two lines below here, added bootstrap noise and removed the " - v"
        A_l = np.dot(inv(joint_ex + N_inv_v_tilde_sim), joint_ex)
        B_l = np.dot(inv(joint_ex + N_inv_v_tilde_sim), (N_inv_XtU_sim + N_inv_w_sim))

        beta_sim_dan_l[n_rv, :] = np.dot(inv(A_l), (beta_hat_priv - B_l))

    for i in range(D):
        empirical_quantile_danl[i][0] = np.percentile(beta_sim_dan_l[:, i], (100 - args.coverage)/2)
        empirical_quantile_danl[i][1] = np.percentile(beta_sim_dan_l[:, i], 100 - (100 - args.coverage)/2)

    return empirical_quantile_danl


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

def private_OLS(X, y, U, beta_true, args):

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
        cov_matrix = sigma_sq_hat_priv * Q_hat_priv
        # check what happens with non PSD Q_hat_priv
        if is_pos_def(cov_matrix) == False:
            cov_matrix = make_pos_def(cov_matrix)
        sqrtN_inv_Z = 1/np.sqrt(N) * np.random.multivariate_normal(np.zeros(D), cov_matrix)   # cecilia -> check this w/ Dan

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


# single trial
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

parser = argparse.ArgumentParser()
parser.add_argument('--num_bootstraps', type=int, default=100, help='Number of experiments to compute quantile')
parser.add_argument('--eps', type=float, default=3, help='privacy threshold args.epsilon')
parser.add_argument('--gamma', type=int, default=5, help='bound on x')
parser.add_argument('--delta', type=float, default=10, help='bound on u')
parser.add_argument('--bound_beta', type=float, default=10.0, help='bound on beta')
parser.add_argument('--z_score', type=float, default=1.96, help='z score associated with confidence level') # 1.96 for 95% confidence level
parser.add_argument('--num_trials', type=int, default=1000, help='Number of experiments to compute coverage ratio')
parser.add_argument('--coverage', type=int, default=95, help='The dimension to be looked at')

parser.add_argument('--test_d', type=int, default=2, help='The dimension to be looked at')
parser.add_argument('--D', type=int, default=4, help='The number of dimensions in the ols problem')
parser.add_argument('--N', type=int, default=100, help='The number of data points in the ols problem')

args = parser.parse_args()
print(args)

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