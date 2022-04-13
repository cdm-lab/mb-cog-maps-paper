"""
Model fitting code for the two stage task, adapted for learning transition matrices on the fly
"""
import numpy as np
import scipy.stats as ss
import pandas as pd
from scipy.special import logsumexp
from math import isclose


def param_init(bounds):
    """
    feed in a list of tuples of bounds and the param_init function will return a set of starting point params from within those bounds
    """
    params = []
    for i in range(len(bounds)):
        lower = bounds[i][0]
        higher = bounds[i][1]
        curr_param = np.random.default_rng().uniform(low=lower, high=higher, size=1)
        params.append(curr_param[0])
    return params


def apply_priors(params, model_func_params):
    """
    returns log-likelihood of parameters you provide based on the prior distribution.
    Priors used taken from Bolenz et al. 2019
    0: softmax inverse temp
    1: learning rate
    2: eligibility trace decay
    3: weight low stakes low arm(this will be the overall weight if the stakes flag is not triggered)
    4: weight high stakes low arm
    5: weight low stakes high arm
    6: weight high stakes high arm
    7: stickiness
    8: response stickiness
    9: eta (transition matrix updating)
    10: kappa (sophisticated updating of the other action)
    """
    LL = MB_MF_rllik_learn_mat_arms(
        params,
        model_func_params[0],
        model_func_params[1],
        model_func_params[2],
        model_func_params[3],
    )
    beta_ll = np.log(ss.gamma.pdf(params[0], 3, scale=0.2))
    alpha_ll = np.log(ss.beta.pdf(params[1], 2, 2))
    lambda_ll = np.log(ss.beta.pdf(params[2], 2, 2))
    wlolo_ll = np.log(ss.beta.pdf(params[3], 2, 2))
    whilo_ll = np.log(ss.beta.pdf(params[4], 2, 2))
    wlohi_ll = np.log(ss.beta.pdf(params[5], 2, 2))
    whihi_ll = np.log(ss.beta.pdf(params[6], 2, 2))

    stick_ll = np.log(ss.norm.pdf(params[7], 0, 1))
    resp_stick_ll = np.log(ss.norm.pdf(params[8], 0, 1))
    if isclose(params[-1], 1.0, rel_tol=1e-8):
        eta_ll = 0
        kappa_ll = 0
    else:
        eta_ll = np.log(ss.beta.pdf(params[9], 2, 2))
        kappa_ll = np.log(ss.beta.pdf(params[10], 2, 2))
    priors_total_ll = (
        beta_ll
        + alpha_ll
        + lambda_ll
        + wlolo_ll
        + whilo_ll
        + wlohi_ll
        + whihi_ll
        + stick_ll
        + resp_stick_ll
        + eta_ll
        + kappa_ll
    )
    posterior_LL = LL - priors_total_ll
    return posterior_LL


def MB_MF_rllik_learn_mat_arms(
    params, sub_df, stakes='4', final=True, kappa_equivalent=False
):
    """
    This function fits a log likelihood for model based versus model free control weights based on the SARSA TD learning algo
    This model acts as though the participant already knows the optimal transition structure
    the params variable should be a list of parameters passed to the model in the following order:
    0: softmax inverse temp
    1: learning rate
    2: eligibility trace decay
    3: weight low stakes low arm(this will be the overall weight if the stakes flag is not triggered)
    4: weight high stakes low arm
    5: weight low stakes high arm
    6: weight high stakes high arm
    7: stickiness
    8: response stickiness
    9: eta (transition matrix updating)
    10: kappa (sophisticated updating of the other action)
    if stakes == 2 the weights for the low arms will be used as weights for the whole task
    """
    b_init = params[0]
    b = b_init
    lr = params[1]
    lamb = params[2]
    w_lo_low_arm = params[
        3
    ]  # if stakes == 2 this will be w_lo/ if stakes == 2c this will be w_low_arm
    w_hi_low_arm = params[4]  # if stakes == 2 this will be w_hi
    w_lo_high_arm = params[5]
    w_hi_high_arm = params[6]  # if stakes == 2c this will be w_high_arm
    st_init = params[7]
    st = st_init
    respst_init = params[8]
    eta = params[9]
    if kappa_equivalent:
        kappa = eta
    else:
        kappa = params[10]
    respst = respst_init
    dtQ = np.zeros((2, 1))
    Qmf = np.ones((4, 2)) * 0.5
    Q2 = np.ones((2, 1)) * 0.5
    if not final:
        Tm = np.zeros((4, 2, 2))
        Tm[0] = [[0.5, 0.5], [0.5, 0.5]]
        Tm[1] = [[0.5, 0.5], [0.5, 0.5]]
        Tm[2] = [[0.5, 0.5], [0.5, 0.5]]
        Tm[3] = [[0.5, 0.5], [0.5, 0.5]]
    else:
        Tm = np.zeros((4, 2, 2))
        Tm[0] = [[1, 0], [0, 1]]
        Tm[1] = [[1, 0], [0, 1]]
        Tm[2] = [[1, 0], [0, 1]]
        Tm[3] = [[1, 0], [0, 1]]
    M = np.zeros((4, 2))
    R = np.asarray([[0], [0]])
    N = len(sub_df.choice1)
    LL = 0

    if sub_df.iloc[0].high_arm == 1:
        environments = [3, 4]
    else:
        environments = [1, 2]
    # loop through the subjects trials
    for trial in range(0, N):

        if (
            sub_df.loc[trial, "rt_1"] == -1 or sub_df.loc[trial, "rt_2"] == -1
        ):  # remove this and screen out rt == -1 in subject csvs instead
            continue

        if sub_df.loc[trial, "stim_left_num"] % 2 == 0:
            R = np.flipud(R)

        s1 = int(sub_df.loc[trial, "state1"])
        s2 = int(sub_df.loc[trial, "state2"])
        a = int(sub_df.loc[trial, "choice1"])
        obj_choice = int(sub_df.loc[trial, "choice1"])
        action = a
        a = a - (s1 - 1) * 2

        Qmb = np.atleast_2d(Tm[s1 - 1]).T @ Q2  # computing model based value function

        if stakes == "4":
            if sub_df.loc[trial, "stake"] == 1 and (
                sub_df.loc[trial, "state1"] not in environments
            ):
                w = w_lo_low_arm
            elif sub_df.loc[trial, "stake"] == 5 and (
                sub_df.loc[trial, "state1"] not in environments
            ):
                w = w_hi_low_arm
            elif sub_df.loc[trial, "stake"] == 1 and (
                sub_df.loc[trial, "state1"] in environments
            ):
                w = w_lo_high_arm
            elif sub_df.loc[trial, "stake"] == 5 and (
                sub_df.loc[trial, "state1"] in environments
            ):
                w = w_hi_high_arm
        elif stakes == "2":
            if sub_df.loc[trial, "stake"] == 1:
                w = w_lo_low_arm
            else:
                w = w_hi_low_arm
        elif stakes == "2c":
            if sub_df.loc[trial, "state1"] not in environments:
                w = w_lo_low_arm
            elif sub_df.loc[trial, "state1"] in environments:
                w = w_hi_high_arm
        elif stakes == "1":
            w = w_lo_low_arm
        else:
            print(stakes)
            raise ValueError("stakes value is not acceptable")

        Q = (
            w * Qmb
            + (1 - w) * np.atleast_2d(Qmf[s1 - 1, :]).T
            + st * np.atleast_2d(M[s1 - 1, :]).T
            + respst * R
        )  # have to do some weird things to change row vecs to column vecs
        # print('Q val is; {0}'.format(Q))
        LL = LL + b * Q[a - 1] - logsumexp(b * Q)
        M[s1 - 1, :] = [0, 0]
        M[s1 - 1, a - 1] = 1

        R = np.zeros((2, 1))
        if obj_choice == sub_df.loc[trial, "stim_left_num"]:
            R[0] = 1
        else:
            R[1] = 1

        # updating transition matrix
        if not final:
            spe = 1 - Tm[s1 - 1, :][s2 - 1, a - 1]
            Tm[s1 - 1][s2 - 1, a - 1] = Tm[s1 - 1][s2 - 1, a - 1] + eta * spe
            Tm[s1 - 1][abs(s2 - 3) - 1, a - 1] = Tm[s1 - 1][abs(s2 - 3) - 1, a - 1] * (
                1 - eta
            )

            virtual_spe = 1 - Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1]
            Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1] = (
                Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1]
                + (eta * kappa) * virtual_spe
            )
            Tm[s1 - 1][s2 - 1, abs(a - 3) - 1] = Tm[s1 - 1][s2 - 1, abs(a - 3) - 1] * (
                1 - (eta * kappa)
            )

        dtQ[0] = Q2[s2 - 1] - Qmf[s1 - 1, a - 1]  # backup with actual choice (SARSA)
        Qmf[s1 - 1, a - 1] = (
            Qmf[s1 - 1, a - 1] + lr * dtQ[0]
        )  # update TD value function

        dtQ[1] = (
            sub_df.loc[trial, "points"] - Q2[s2 - 1]
        )  # prediction error for 2nd choice

        Q2[s2 - 1] = Q2[s2 - 1] + lr * dtQ[1]  # updated TD value function
        Qmf[s1 - 1, a - 1] = (
            Qmf[s1 - 1, a - 1] + lamb * lr * dtQ[1]
        )  # eligibility trace
    return -LL
