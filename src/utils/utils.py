"""
Utils functions for the memory two step task data munging
Functions:
create_sub_dict(subnum,df)
create_fit_df(subnum,df)
create_sub_background_df(sub_dict,subnum,df)
create_sub_sliders_df(sub_dict,subnum,df)
create_state1_sorted_sliders(sub_dict,subnum,df)
create_state2_sorted_sliders(sub_dict,subnum,df)
create_state2_high_arm_sorted_sliders(sub_dict,subnum,df)
"""

import numpy as np


def create_sub_dict(subnum, df):
    """
    Generate a subject specific dict for use in other functions
    """
    sub_dict = {}
    sub_df = df[df['subid'] == subnum]
    if len(sub_df) < 256:  # checks to see if subject has completed all trials (if they don't respond they are still
        # counted towards length)
        return 'is a bad subject'
    sub_df['response1'].replace(70, 1, inplace=True)
    sub_df['response1'].replace(74, 2, inplace=True)
    sub_df['response1'].replace('f', 1, inplace=True)
    sub_df['response1'].replace('j', 2, inplace=True)
    if len(set([1, 2]) - set(
            sub_df['response1'].unique())) != 0:  # checks to see if subject hasn't responded with both keys
        return 'is a bad subject'
    if sub_df['high_arm'].unique()[0] == 1:
        sub_dict['high_arm'] = 'inside'
    else:
        sub_dict['high_arm'] = 'outside'
    envs = ['desert', 'forest', 'library', 'restaurant']
    object_list = []

    for i in [1, 2, 3, 4]:
        sub_dict[envs[i - 1]] = sub_df[sub_df['state1'] == i].iloc[0][['stim_left', 'stim_right']].values
        object_list.append(list(sub_dict[envs[i - 1]]))
    flat_list = [item for sublist in object_list for item in sublist]
    sub_dict['object_list'] = flat_list
    condensed = sub_df[['state1', 'stim_left', 'stim_right', 'response1', 'state2']]
    condensed = condensed.drop_duplicates()
    condensed = condensed.sort_values(by=['state1', 'response1'])
    rew_1 = []
    rew_2 = []
    for i in range(len(condensed)):
        if condensed.iloc[i].response1 == -1:
            continue

        if (condensed.iloc[i].response1 == 1) and (condensed.iloc[i].state2 == '1'):
            rew_1.append(condensed.iloc[i]['stim_left'])
            rew_2.append(condensed.iloc[i]['stim_right'])
        elif (condensed.iloc[i].response1 == 1) and (condensed.iloc[i].state2 == '2'):
            rew_2.append(condensed.iloc[i]['stim_left'])
            rew_1.append(condensed.iloc[i]['stim_right'])
    sub_dict['reward_1'] = list(
        dict.fromkeys(rew_1))  # grabs every other index and is sorted by desert,forest,library,restaurant
    sub_dict['reward_2'] = list(dict.fromkeys(rew_2))
    sub_dict['task_backgrounds'] = list(sub_df.background.unique())
    check = sub_df.groupby(['state2'])['stim_state2'].unique()
    if '1' not in check.index:
        return 'is a bad subject'
    elif '2' not in check.index:
        return 'is a bad subject'

    sub_dict['reward_1_object'] = check.loc['1'][0]
    sub_dict['reward_2_object'] = check.loc['2'][0]

    object_dict = {}
    rew_objs = [check.loc['1'][0], check.loc['2'][0]]

    for x in flat_list:
        if x not in rew_objs:
            env_name = envs[sub_df[sub_df['stim_left'] == x].iloc[0]['state1'] - 1]
            object_dict[x] = env_name
    sub_dict['object_dict'] = object_dict
    return sub_dict


def create_fit_df(subnum, df):
    """
    Generate a subject specific df for use in model fitting
    """
    sub_df = df[df['subid'] == subnum].reset_index(drop=True)
    sub_df['response1'].replace(70, 1, inplace=True)
    sub_df['response1'].replace(74, 2, inplace=True)
    sub_df['response1'].replace('f', 1, inplace=True)
    sub_df['response1'].replace('j', 2, inplace=True)
    condensed = sub_df[['state1', 'stim_left', 'stim_right', 'response1', 'choice1']]
    condensed = condensed.drop_duplicates()
    condensed = condensed.sort_values(by=['state1', 'response1'])
    new_map_dict = {}
    for i in range(len(condensed)):
        if condensed.iloc[i].response1 == -1:
            continue
        if condensed.iloc[i].response1 == 1:
            new_map_dict[condensed.iloc[i].stim_left] = condensed.iloc[i].choice1
        elif condensed.iloc[i].response1 == 2:
            new_map_dict[condensed.iloc[i].stim_right] = condensed.iloc[i].choice1
    sub_df.loc[:, 'stim_left_num'] = sub_df['stim_left']
    sub_df['stim_left_num'].replace(new_map_dict, inplace=True)
    sub_df.loc[:, 'stim_right_num'] = sub_df['stim_right']
    sub_df['stim_right_num'].replace(new_map_dict, inplace=True)
    sub_df.loc[:, 'points'] = sub_df['points'].apply(lambda x: x / 9)
    # sub_df = sub_df.drop(columns={'index'})
    return sub_df


def create_state1_sorted_sliders(sub_dict, subnum, df):
    presliders = df[df['prepost'] == 'pre']
    postsliders = df[df['prepost'] == 'post']
    presliders = presliders.drop_duplicates()
    object_ordered_list = []
    object_ordered_list.append(list(sub_dict[subnum]['reward_1']))
    rew1 = sub_dict[subnum]['reward_1_object']
    object_ordered_list.append([rew1])
    object_ordered_list.append(list(sub_dict[subnum]['reward_2']))
    rew2 = sub_dict[subnum]['reward_2_object']
    object_ordered_list.append([rew2])
    flat_list = [item for sublist in object_ordered_list for item in sublist]

    pre_sorted_identity = np.eye(10) * 100
    for i in range(len(presliders)):
        object_1 = presliders.iloc[i]['stim1'][8:-4]
        object_2 = presliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_list.index(object_1)
        idx2 = flat_list.index(object_2)
        pre_sorted_identity[idx1, idx2] = presliders['response'].iloc[i]
    post_sorted_identity = np.eye(10) * 100
    for i in range(len(postsliders)):
        object_1 = postsliders.iloc[i]['stim1'][8:-4]
        object_2 = postsliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_list.index(object_1)
        idx2 = flat_list.index(object_2)
        post_sorted_identity[idx1, idx2] = postsliders['response'].iloc[i]
    # adding the upper lower triangle averaging for pre_sorted
    lower = np.tril_indices(10)
    averaged = (pre_sorted_identity[lower] + pre_sorted_identity.T[lower]) / 2
    averaged_no_nan = averaged.copy()
    for i in range(len(averaged)):
        if np.isnan(averaged[i]):
            if np.isnan(pre_sorted_identity[lower][i]) & ~np.isnan(pre_sorted_identity.T[lower][i]):
                averaged_no_nan[i] = pre_sorted_identity.T[lower][i]
            elif not np.isnan(pre_sorted_identity[lower][i]):
                averaged_no_nan[i] = pre_sorted_identity[lower][i]
            else:
                averaged_no_nan[i] = 0
    x = np.eye(10) * 100
    x[lower] = averaged_no_nan
    x.T[lower] = averaged_no_nan
    # adding the upper lower triangle for post_sorted
    averaged = (post_sorted_identity[lower] + post_sorted_identity.T[lower]) / 2
    averaged_no_nan = averaged.copy()
    for i in range(len(averaged)):
        if np.isnan(averaged[i]):
            if np.isnan(post_sorted_identity[lower][i]) & ~np.isnan(post_sorted_identity.T[lower][i]):
                averaged_no_nan[i] = post_sorted_identity.T[lower][i]
            elif not np.isnan(post_sorted_identity[lower][i]):
                averaged_no_nan[i] = post_sorted_identity[lower][i]
            else:
                averaged_no_nan[i] = 0
    y = np.eye(10) * 100
    y[lower] = averaged_no_nan
    y.T[lower] = averaged_no_nan

    return [x, y]


def create_stake_sorted_sliders(sub_dict, subnum, df):
    """
    creates stake sorted sliders
    """
    isbad = False
    presliders = df[df['prepost'] == 'pre']
    postsliders = df[df['prepost'] == 'post']
    presliders = presliders.drop_duplicates()
    assert 'high_arm' in sub_dict[subnum].keys(), f'{sub_dict[subnum]}'
    if sub_dict[subnum]['high_arm'] == 'outside':
        rew_1_obj_list = list(sub_dict[subnum]['reward_1'])
        rew_2_obj_list = list(sub_dict[subnum]['reward_2'])
    else:
        rew_1_unsort = list(sub_dict[subnum]['reward_1'])
        rew_2_unsort = list(sub_dict[subnum]['reward_2'])
        rew_1_obj_list = [rew_1_unsort[2], rew_1_unsort[3], rew_1_unsort[0], rew_1_unsort[1]]
        rew_2_obj_list = [rew_2_unsort[2], rew_2_unsort[3], rew_2_unsort[0], rew_2_unsort[1]]
    object_ordered_list = []
    object_ordered_list.append(rew_1_obj_list)
    rew1 = sub_dict[subnum]['reward_1_object']
    object_ordered_list.append([rew1])
    object_ordered_list.append(rew_2_obj_list)
    rew2 = sub_dict[subnum]['reward_2_object']
    object_ordered_list.append([rew2])
    flat_list = [item for sublist in object_ordered_list for item in sublist]

    pre_sorted_identity = np.eye(10) * 100
    for i in range(len(presliders)):
        object_1 = presliders.iloc[i]['stim1'][8:-4]
        object_2 = presliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_list.index(object_1)
        idx2 = flat_list.index(object_2)
        pre_sorted_identity[idx1, idx2] = presliders['response'].iloc[i]
    post_sorted_identity = np.eye(10) * 100
    for i in range(len(postsliders)):
        object_1 = postsliders.iloc[i]['stim1'][8:-4]
        object_2 = postsliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_list.index(object_1)
        idx2 = flat_list.index(object_2)
        post_sorted_identity[idx1, idx2] = postsliders['response'].iloc[i]
    # adding the upper lower triangle averaging for pre_sorted
    lower = np.tril_indices(10)
    averaged = (pre_sorted_identity[lower] + pre_sorted_identity.T[lower]) / 2
    averaged_no_nan = averaged.copy()
    for i in range(len(averaged)):
        if np.isnan(averaged[i]):
            if np.isnan(pre_sorted_identity[lower][i]) & ~np.isnan(pre_sorted_identity.T[lower][i]):
                averaged_no_nan[i] = pre_sorted_identity.T[lower][i]
            elif not np.isnan(pre_sorted_identity[lower][i]):
                averaged_no_nan[i] = pre_sorted_identity[lower][i]
            else:
                averaged_no_nan[i] = 0
                isbad = True
    x = np.eye(10) * 100
    x[lower] = averaged_no_nan
    x.T[lower] = averaged_no_nan
    # adding the upper lower triangle for post_sorted
    averaged = (post_sorted_identity[lower] + post_sorted_identity.T[lower]) / 2
    averaged_no_nan = averaged.copy()
    for i in range(len(averaged)):
        if np.isnan(averaged[i]):
            if np.isnan(post_sorted_identity[lower][i]) & ~np.isnan(post_sorted_identity.T[lower][i]):
                averaged_no_nan[i] = post_sorted_identity.T[lower][i]
            elif not np.isnan(post_sorted_identity[lower][i]):
                averaged_no_nan[i] = post_sorted_identity[lower][i]
            else:
                averaged_no_nan[i] = 0
                isbad = True
    y = np.eye(10) * 100
    y[lower] = averaged_no_nan
    y.T[lower] = averaged_no_nan

    return [x, y], isbad


def create_state2_sorted_sliders(sub_dict, subnum, df):
    presliders = df[df['prepost'] == 'pre']
    postsliders = df[df['prepost'] == 'post']
    presliders = presliders.drop_duplicates()
    second_ordered_list = []
    rew1 = sub_dict[subnum]['reward_1_object']
    rew2 = sub_dict[subnum]['reward_2_object']
    second_ordered_list.append(list(sub_dict[subnum]['reward_1']))  # have to reverse order so high arm is first
    second_ordered_list.append([rew2])
    second_ordered_list.append(list(sub_dict[subnum]['reward_2']))
    second_ordered_list.append([rew1])
    flat_reward_list = [item for sublist in second_ordered_list for item in sublist]

    pre_reward_identity = np.eye(10) * 100
    for i in range(len(presliders)):
        object_1 = presliders.iloc[i]['stim1'][8:-4]
        object_2 = presliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_reward_list.index(object_1)
        idx2 = flat_reward_list.index(object_2)
        pre_reward_identity[idx1, idx2] = presliders['response'].iloc[i]

    post_reward_identity = np.eye(10) * 100
    for i in range(len(postsliders)):
        object_1 = postsliders.iloc[i]['stim1'][8:-4]
        object_2 = postsliders.iloc[i]['stim2'][8:-4]
        idx1 = flat_reward_list.index(object_1)
        idx2 = flat_reward_list.index(object_2)
        post_reward_identity[idx1, idx2] = postsliders['response'].iloc[i]
    return [pre_reward_identity, post_reward_identity]


def create_model_matrices():
    """
    creates model matrices for connections based on second stage rewards or connections based on initial nodes being
    shared
    """
    post_state1_sort = np.eye(10)

    post_state1_sort[0, 5] = 1
    post_state1_sort[5, 0] = 1

    post_state1_sort[1, 6] = 1
    post_state1_sort[6, 1] = 1

    post_state1_sort[2, 7] = 1
    post_state1_sort[7, 2] = 1

    post_state1_sort[3, 8] = 1
    post_state1_sort[8, 3] = 1

    post_state1_sort[9, 4] = .5
    post_state1_sort[4, 9] = .5

    post_state2_sort = np.eye(10)

    for i in range(5):
        for j in range(5):
            post_state2_sort[i, j] = 1
            post_state2_sort[j, i] = 1
            post_state2_sort[i + 5, j + 5] = 1
            post_state2_sort[j + 5, i + 5] = 1
    l_sort = np.eye(10)
    for i in range(5):
        l_sort[i, 4] = 1
        l_sort[4, i] = 1
        l_sort[i + 5, 9] = 1
        l_sort[9, i + 5] = 1
    o_sort = np.eye(10)
    for i in range(4):
        for j in range(4):
            o_sort[i, j] = 1
            o_sort[j, i] = 1
            o_sort[i + 5, j + 5] = 1
            o_sort[j + 5, i + 5] = 1
    return [post_state1_sort, post_state2_sort, l_sort, o_sort]


def create_model_matrices_hvl():
    """
    :return: model_matrix dictionary containing the 3 high vs low model matrices
    """
    model_mats_highlow = {'o_high': np.eye(10)}
    model_mats_highlow['o_high'][0, 1] = 1
    model_mats_highlow['o_high'][1, 0] = 1
    model_mats_highlow['o_high'][5, 6] = 1
    model_mats_highlow['o_high'][6, 5] = 1

    model_mats_highlow['o_low'] = np.eye(10)
    model_mats_highlow['o_low'][2, 3] = 1
    model_mats_highlow['o_low'][3, 2] = 1
    model_mats_highlow['o_low'][7, 8] = 1
    model_mats_highlow['o_low'][8, 7] = 1

    model_mats_highlow['l_low'] = np.eye(10)
    model_mats_highlow['l_low'][4, 2] = 1
    model_mats_highlow['l_low'][2, 4] = 1
    model_mats_highlow['l_low'][4, 3] = 1
    model_mats_highlow['l_low'][3, 4] = 1
    model_mats_highlow['l_low'][9, 7] = 1
    model_mats_highlow['l_low'][7, 9] = 1
    model_mats_highlow['l_low'][9, 8] = 1
    model_mats_highlow['l_low'][8, 9] = 1

    model_mats_highlow['l_high'] = np.eye(10)
    model_mats_highlow['l_high'][4, 0] = 1
    model_mats_highlow['l_high'][0, 4] = 1
    model_mats_highlow['l_high'][4, 1] = 1
    model_mats_highlow['l_high'][1, 4] = 1
    model_mats_highlow['l_high'][9, 5] = 1
    model_mats_highlow['l_high'][5, 9] = 1
    model_mats_highlow['l_high'][9, 6] = 1
    model_mats_highlow['l_high'][6, 9] = 1

    post_state1_sort = np.eye(10)
    model_mats_highlow['state1_low'] = post_state1_sort.copy()
    model_mats_highlow['state1_high'] = post_state1_sort.copy()

    model_mats_highlow['state1_high'][0, 5] = 1
    model_mats_highlow['state1_high'][5, 0] = 1

    model_mats_highlow['state1_high'][1, 6] = 1
    model_mats_highlow['state1_high'][6, 1] = 1

    model_mats_highlow['state1_low'][2, 7] = 1
    model_mats_highlow['state1_low'][7, 2] = 1

    model_mats_highlow['state1_low'][3, 8] = 1
    model_mats_highlow['state1_low'][8, 3] = 1

    return model_mats_highlow