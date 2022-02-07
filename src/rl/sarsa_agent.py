"""
Model Based/Model Free agent (adapted from Wouter Kool's tradeoff code)
Ata Karagoz Feb. 2021 based on code written by Wouter Kool for tradeoffs
"""

import numpy as np


def create_task_lists(N, rewards, stakes, high_arm):
    """
    Generates task list for use with the stimulation
    N: number of trials to simulate
    rewards: specify the bounds and sd of the reward walk
    stakes: a tuple of stake distributions for the high arm (i.e. high stake first)
    arms: specify which state is high arm
    random walk code adapted from Jack Dolgin
    """

    if high_arm == 1:
        environments = [3, 4]
    else:
        environments = [1, 2]

    total_n = np.zeros((N, 1))
    quarters = len(total_n) / 4
    for i in range(4):
        initial = int(quarters * i)
        end = int(quarters * (i + 1))
        total_n[initial:end] = i + 1
    stake_list = np.ones((N, 1))
    for i in range(N):
        if total_n[i] in environments:
            if np.random.uniform(0, 1) < stakes[0]:
                stake_list[i] = 5
        else:
            if np.random.uniform(0, 1) < stakes[1]:
                stake_list[i] = 5
    mean = rewards[0]
    sd = rewards[1]
    rwalk_min = 0
    rwalk_max = 9
    rew1 = np.zeros((N, 1))  # make reward random walk
    for i in range(N):
        change_by = np.random.normal(mean, sd, 1)
        rew1[i] = (rew1[i] + change_by[0]).round()

        while not rwalk_min <= rew1[i] <= rwalk_max:
            if rew1[i] > rwalk_max:
                rew1[i] -= 2 * (rew1[i] - rwalk_max)
            elif rew1[i] < rwalk_min:
                rew1[i] += 2 * (rwalk_min - rew1[i])

    rew2 = np.zeros((N, 1))  # make reward 2 random walk
    for i in range(N):
        change_by = np.random.normal(mean, sd, 1)
        rew2[i] = (rew2[i] + change_by[0]).round()

        while not rwalk_min <= rew2[i] <= rwalk_max:
            if rew2[i] > rwalk_max:
                rew2[i] -= 2 * (rew2[i] - rwalk_max)
            elif rew2[i] < rwalk_min:
                rew2[i] += 2 * (rwalk_min - rew2[i])

    rews = np.hstack((rew1, rew2))
    state_and_stake = np.hstack((total_n, stake_list))
    np.random.shuffle(state_and_stake)
    return rews.astype(int), state_and_stake.astype(int)


def MBMF_deterministic_1choice_rew_sim(params, rews, states, stakes, final, high_arm, w_count, coin_flip):
    """
    Mixed model-based / model-free simulation code for a task with
    deterministic transitions, one choice at the second level, and points for
    the second-level bandits.
     USAGE: output = MBMF_deterministic_1choice_rew_sim(x,rews)
     INPUTS:
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
     rews: [N x 2] array storing the rewards, where
            rews(n,s) is the payoff on trial n in second-level state s after
            taking action a, where N is the number of trials
     states: [N x 1] array storing the initial states for the model env
     stakes: [N x 1] array storing the stake values for the model env
      OUTPUTS:
        actions: [N x 1] chosen actions at first level
        rewards: [N x 1] second level rewards
        states: [N x 2] first and second level states
    """
    # parameters
    # global w
    b_init = params[0]
    b = b_init
    lr = params[1]
    lamb = params[2]
    w_lo_low_arm = params[3]  # if stakes == 2 this will be w_lo
    w_hi_low_arm = params[4]  # if stakes == 2 this will be w_hi
    w_lo_high_arm = params[5]
    w_hi_high_arm = params[6]
    st_init = params[7]
    st = st_init
    respst_init = params[8]
    eta = params[9]
    kappa = params[10]
    respst = respst_init

    # initializing
    N = rews.shape[0]
    actions = np.zeros((N, 1))
    rewards = np.zeros((N, 1))
    states_traversed = np.zeros((N, 2))

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

    if high_arm == 1:
        environments = [3, 4]
    else:
        environments = [1, 2]

    # running through trials
    for t in range(N):
        s1 = states[t]
        Qmb = np.atleast_2d(Tm[s1 - 1]).T @ Q2  # compute model-based value function

        if w_count == 4:
            if stakes[t] == 1 and (s1 not in environments):
                w = w_lo_low_arm
            elif stakes[t] == 5 and (s1 not in environments):
                w = w_hi_low_arm
            elif stakes[t] == 1 and (s1 in environments):
                w = w_lo_high_arm
            elif stakes[t] == 5 and (s1 in environments):
                w = w_hi_high_arm
        elif w_count == 2:
            if stakes[t] == 1:
                w = w_lo_low_arm
            else:
                w = w_hi_low_arm
        else:
            w = w_lo_low_arm
        # Q = w*Qmb + (1-w)*np.atleast_2d(Qmf[s1-1,:]).T         # mixing model based and model free
        Q = w * Qmb + (1 - w) * np.atleast_2d(Qmf[s1 - 1, :]).T + st * np.atleast_2d(M[s1 - 1, :]).T + respst * R
        if coin_flip is None:
            if np.random.uniform(0, 1) > np.exp(b * Q[1]) / sum(np.exp(b * Q)):  # make choice using softmax
                a = 1
            else:
                a = 2
        else:
            if np.random.uniform(0, 1) < 0.5:  # make choice using coin flip
                a = 1
            else:
                a = 2

        s2 = a
        M[s1 - 1, :] = [0, 0]
        M[s1 - 1, a - 1] = 1

        R = np.zeros((2, 1))
        if a == 1:
            R[0] = 1
        else:
            R[1] = 1

        if not final:  # updating transition matrix
            spe = 1 - Tm[s1 - 1, :][s2 - 1, a - 1]
            Tm[s1 - 1][s2 - 1, a - 1] = Tm[s1 - 1][s2 - 1, a - 1] + eta * spe
            Tm[s1 - 1][abs(s2 - 3) - 1, a - 1] = Tm[s1 - 1][abs(s2 - 3) - 1, a - 1] * (1 - eta)

            virtual_spe = 1 - Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1]
            Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1] = Tm[s1 - 1][abs(s2 - 3) - 1, abs(a - 3) - 1] + (
                    eta * kappa) * virtual_spe
            Tm[s1 - 1][s2 - 1, abs(a - 3) - 1] = Tm[s1 - 1][s2 - 1, abs(a - 3) - 1] * (1 - (eta * kappa))

        dtQ[0] = Q2[s2 - 1] - Qmf[s1 - 1, a - 1]  # backup with actual choice (i.e., sarsa)
        Qmf[s1 - 1, a - 1] = Qmf[s1 - 1, a - 1] + lr * dtQ[0]  # update TD value function

        dtQ[1] = rews[t, s2 - 1] - Q2[s2 - 1]  # prediction error (2nd choice)

        Q2[s2 - 1] = Q2[s2 - 1] + lr * dtQ[1]  # update TD value function
        Qmf[s1 - 1, a - 1] = Qmf[s1 - 1, a - 1] + lamb * lr * dtQ[1]  # eligibility trace

        actions[t] = a
        rewards[t] = rews[t, s2 - 1]
        states_traversed[t, :] = [s1, s2]

    return actions, rewards, states_traversed