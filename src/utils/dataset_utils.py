import utils
import pandas as pd
import pickle
import os
from sklearn import linear_model
import numpy as np
from scipy import stats
from sklearn import decomposition


def stringify(input_df):
    """
    Returns in place a version of the dataframe where the subject id column is changed to a str type if it isn't already
    :param input_df: dataframe you want to change
    :return: nothing (alters df in place)
    """
    if not isinstance(input_df.loc[0, 'subid'], str):
        input_df['subid'] = input_df['subid'].astype(str)


def filter_df(input_df, subject_list):
    """
    Returns a filtered version of the input dataframe where subjects that aren't in the list are removed
    :param input_df: any of the interim dataframes
    :param subject_list: list of subjects you want to select
    :return: filtered_df which contains only subjects you want
    """
    if 'Unnamed: 0' in input_df.columns:
        input_df = input_df.drop(columns={'Unnamed: 0'})
    filtered_df = input_df[input_df['subid'].isin(subject_list)]
    return filtered_df


def make_sub_dicts(input_path):
    """
    Generates subject dictionaries containing information about which contexts were high-stake, which objects belong to
    which context, etc.
    :param input_path: path where the data live and where the sub_dict.pickle will be saved
    :return:nothing
    """
    stake_df = pd.read_csv(os.path.join(input_path, 'memory_2step_stake_data.csv'))
    stringify(stake_df)
    sub_dict = {}

    for sub in stake_df.subid.unique():
        sub_dict[sub] = utils.create_sub_dict(sub, stake_df)

    output_pickle = os.path.join(input_path, 'sub_dicts.pickle')
    with open(output_pickle, 'wb') as handle:
        pickle.dump(sub_dict, handle)


def make_good_subjects_list(input_path, output_path, exp_num=2):
    stake_df = pd.read_csv(os.path.join(input_path, 'memory_2step_stake_data.csv'))
    slider_df = pd.read_csv(os.path.join(input_path, 'memory_2step_slider_data.csv'))
    slider_df = slider_df.drop_duplicates()
    back_df = pd.read_csv(os.path.join(input_path, 'memory_2step_background_data.csv'))
    back_df['rt'].fillna(-1, inplace=True)
    pickle_path = os.path.join(input_path, 'sub_dicts.pickle')
    with open(pickle_path, 'rb') as handle:
        sub_dict = pickle.load(handle)

    for df in stake_df, slider_df, back_df:
        stringify(df)

    num_thresh_stake = 0
    num_thresh_dict = 0
    num_thresh_slider = 0
    num_thresh_back = 0
    bad_subjects = ['smorse', 'atabk', 'A2R75YFKVALBXE', 'A1ROEDVMTO9Y3X', 'A1T3ROSW2LC4FG', 'A34CPKFZXBX1PO',
                    'A3LVLZS8S41ZD7', 'A1Y0Y6U906ABT5', 'A26RO8GGTQAXGG', 'A1FKRZKU1H9YFC']  # test participants
    subids = stake_df.subid.unique()
    for sub in subids:  # create bad_subjects list based on RT threshold
        percentages = stake_df[stake_df['subid'] == sub]['rt_2'].value_counts(normalize=True) * 100
        sub_slider_df = slider_df[slider_df['subid'] == sub]
        if sub_dict[sub] == 'is a bad subject':
            bad_subjects.append(sub)
            num_thresh_dict += 1
            continue
        if -1 in percentages.index:
            if percentages[-1] > 20:
                bad_subjects.append(sub)
                num_thresh_stake += 1
                continue
        # Cleaning based on behRSA task
        if len(sub_slider_df) != 180:
            bad_subjects.append(sub)
            num_thresh_slider += 1
            continue
        if sub_slider_df['response'].isna().sum() >= 18:
            bad_subjects.append(sub)
            num_thresh_slider += 1
            continue
        stake, isbad = utils.create_stake_sorted_sliders(sub_dict, sub, sub_slider_df)
        if isbad:
            bad_subjects.append(sub)
            num_thresh_slider += 1
            continue
        # Cleaning based on memory_df
        if exp_num == 2:
            percentages_mem = back_df[back_df['subid'] == sub]['rt'].value_counts(normalize=True) * 100
            if -1 in percentages_mem.index:
                if percentages_mem[-1] > 20:
                    bad_subjects.append(sub)
                    num_thresh_back += 1
                    continue

    good_subjects = [x for x in subids if x not in bad_subjects]
    print(f'thresholded by dict {num_thresh_dict}')
    print(f'thresholded by stakes {num_thresh_stake}')
    print(f'thresholded by slider {num_thresh_slider}')
    print(f'thresholded by memory {num_thresh_back}')
    filename = os.path.join(output_path, f'exp{exp_num}_good_subjects.txt')
    with open(filename, 'w') as f:
        for sub in good_subjects:
            f.write(f'{sub}\n')
    return good_subjects


def model_fit_threshold(input_path, output_path, subjects, exp_num):
    w1_map_df = pd.read_csv(os.path.join(input_path, 'w1_map_df.csv'))
    bad_subjects = []
    num_thresh_model_fit = 0
    subids = subjects
    for sub in subids:
        sub_w1_map_df = w1_map_df[w1_map_df['subid'] == sub]
        assert len(sub_w1_map_df) > 0, f'you need to rerun model fit for {sub}!'
        if sub_w1_map_df['LL'].values[0] > 172.846:
            bad_subjects.append(sub)
            num_thresh_model_fit += 1
            continue
    print(f'thresholded by model fit {num_thresh_model_fit}')
    good_subjects = [x for x in subids if x not in bad_subjects]
    filename = os.path.join(output_path, f'exp{exp_num}_good_subjects.txt')
    with open(filename, 'w') as f:
        for sub in good_subjects:
            f.write(f'{sub}\n')
    return good_subjects


def make_overall_stake_df(input_path, output_path, subjects=None):
    """
    Creates overall_stake_df given the input_path, output_path, and subject_list
    :param input_path: path to look for the raw csv data for the 2step task
    :param output_path: path to save out the overall stake_df and sub_dict
    :param subjects: the subids that you would like to include
    :return: outputs overall_stake_df to output_path
    """

    stake_df = pd.read_csv(os.path.join(input_path, 'memory_2step_stake_data.csv'))
    stringify(stake_df)
    subids = subjects
    assert len(set(subids) - set(stake_df.subid.unique())) <= 0, "trying to grab subjects that don't exist in raw data!"
    overall_df = pd.DataFrame()
    for sub in subids:
        sub_df = utils.create_fit_df(sub, stake_df)
        filename = f'subject_csvs/sub_{sub}.csv'
        out_file = os.path.join(output_path, filename)
        sub_df.to_csv(out_file)
        overall_df = pd.concat([overall_df, sub_df])
    # fixing stake df
    overall_df['high or low arm'] = 'low'
    overall_df = overall_df.reset_index()
    overall_df.loc[np.where((overall_df['high_arm'] == 2) & (overall_df['state1'] == 1))[0], 'high or low arm'] = 'high'
    overall_df.loc[np.where((overall_df['high_arm'] == 2) & (overall_df['state1'] == 2))[0], 'high or low arm'] = 'high'

    overall_df.loc[np.where((overall_df['high_arm'] == 1) & (overall_df['state1'] == 3))[0], 'high or low arm'] = 'high'
    overall_df.loc[np.where((overall_df['high_arm'] == 1) & (overall_df['state1'] == 4))[0], 'high or low arm'] = 'high'
    overall_out_file = os.path.join(output_path, 'overall_stake_df.csv')
    overall_df = overall_df.drop(columns={'index'})
    overall_df.to_csv(overall_out_file)


def make_slider_df(input_path, output_path, subjects=None, hvl=False):
    """
    Generates the dataframe and slider_dicts that contains the behRSA data
    :param input_path: path to look for the raw csv data for the slider task
    :param output_path: path to save out
    slider_dict and slider_df
    :param subjects: list of subjects to include in analysis (if not provided subids are
    sourced from sub_dict in input path)
    :param hvl: whether you want standard model matrices or are comparing hvl (hvl=True)
    :return: outputs a slider_dict and slider_df containing data for use in further analyses
    """
    slider_df = pd.read_csv(os.path.join(input_path, 'memory_2step_slider_data.csv'))
    slider_df = slider_df.drop_duplicates()
    stringify(slider_df)
    pickle_path = os.path.join(input_path, 'sub_dicts.pickle')
    with open(pickle_path, 'rb') as handle:
        sub_dict = pickle.load(handle)
    slider_dict = {}
    subids = subjects
    assert len(
        set(subids) - set(slider_df.subid.unique())) <= 0, "trying to grab subjects that don't exist in raw data!"
    for sub in subids:
        sub_slider_df = slider_df[slider_df['subid'] == sub]
        stake, isbad = utils.create_stake_sorted_sliders(sub_dict, sub, sub_slider_df)
        slider_dict[sub] = [sub, stake]
    coef_vals = {}
    if not hvl:
        model_mats = utils.create_model_matrices()
        # Fitting standard version of matrices
        for sub in slider_dict:
            post_minus_pre_state1 = slider_dict[sub][1][1] - slider_dict[sub][1][0]
            np.fill_diagonal(post_minus_pre_state1, 100)
            sub_regr_df = pd.DataFrame()
            lower_indices = np.tril_indices(10, -1, 10)  # m, k n
            sub_regr_df['state1'] = model_mats[0][lower_indices].flatten()
            sub_regr_df['l'] = model_mats[2][lower_indices].flatten()
            sub_regr_df['o'] = model_mats[3][lower_indices].flatten()
            sub_regr_df['subject_data'] = post_minus_pre_state1[lower_indices].flatten()
            sub_regr_df['subid'] = sub
            regr_high = linear_model.LinearRegression()
            X = sub_regr_df[['state1', 'l', 'o']]
            y = sub_regr_df['subject_data']
            regr_high.fit(X, y)
            # adj_rsquared = 1 - (1-regr_high.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1) #calculating adju1sted_rsquared
            coef_vals[sub] = [regr_high.coef_[0], regr_high.coef_[1], regr_high.coef_[2], sub_dict[sub]['high_arm']]
        model_mat_fits = pd.DataFrame(coef_vals).T.reset_index()
        model_mat_fits = model_mat_fits.rename(
            columns={'index': 'subid', 0: 'state1_coef', 1: 'l_coef', 2: 'o_coef', 3: 'high_arm'})
        model_mat_outfile = os.path.join(output_path, 'model_mat_fits.csv')
    if hvl:
        model_mats_highlow = utils.create_model_matrices_hvl()
        coef_vals = {}
        for sub in slider_dict:
            post_minus_pre_state1 = slider_dict[sub][1][1] - slider_dict[sub][1][0]
            np.fill_diagonal(post_minus_pre_state1, 100)
            sub_regr_df = pd.DataFrame()
            lower_indices = np.tril_indices(10, -1, 10)  # m, k n
            sub_regr_df['state1_high'] = model_mats_highlow['state1_high'][lower_indices].flatten()
            sub_regr_df['state1_low'] = model_mats_highlow['state1_low'][lower_indices].flatten()
            sub_regr_df['l_high'] = model_mats_highlow['l_high'][lower_indices].flatten()
            sub_regr_df['l_low'] = model_mats_highlow['l_low'][lower_indices].flatten()
            sub_regr_df['o_high'] = model_mats_highlow['o_high'][lower_indices].flatten()
            sub_regr_df['o_low'] = model_mats_highlow['o_low'][lower_indices].flatten()
            sub_regr_df['subject_data'] = post_minus_pre_state1[lower_indices].flatten()
            sub_regr_df['subid'] = sub
            regr_high = linear_model.LinearRegression()
            regr_high.fit(sub_regr_df[['state1_high', 'l_high', 'o_high', 'state1_low', 'l_low', 'o_low']],
                          sub_regr_df['subject_data'])
            coef_vals[sub] = [regr_high.coef_[0], regr_high.coef_[1], regr_high.coef_[2],
                              regr_high.coef_[3], regr_high.coef_[4], regr_high.coef_[5], sub_dict[sub]['high_arm']]
            model_mat_fits = pd.DataFrame(coef_vals).T.reset_index()
            model_mat_fits = model_mat_fits.rename(
                columns={'index': 'subid', 0: 'state1_high_coef', 1: 'l_high_coef', 2: 'o_high_coef',
                         3: 'state1_low_coef', 4: 'l_low_coef', 5: 'o_low_coef', 6: 'high_arm'})
            model_mat_fits['state1_diff'] = model_mat_fits['state1_high_coef'] - model_mat_fits['state1_low_coef']
            model_mat_fits['l_diff'] = model_mat_fits['l_high_coef'] - model_mat_fits['l_low_coef']
            model_mat_fits['o_diff'] = model_mat_fits['o_high_coef'] - model_mat_fits['o_low_coef']
            model_mat_outfile = os.path.join(output_path, 'model_mat_fits_hvl.csv')
    model_mat_fits.to_csv(model_mat_outfile)
    slider_pickle_out = os.path.join(output_path, 'slider_dicts.pickle')
    pickle.dump(slider_dict, open(slider_pickle_out, "wb"))


def make_memory_df(input_path, output_path, subjects=None, exp_num=2):
    """
    Makes the memory dataframe and dprime dataframe for subjects from exp2
    :param input_path: path to look for data and sub_dicts.pickle file
    :param output_path: path to save out dprime_df, and memory_df
    :param subjects: list of subjects to include in analysis (if not provided subids are
    sourced from sub_dict in input path)
    :param exp_num: which experiment version you are using (default is 2)
    :return: outputs a dprime_df and memory_df
    """
    # requires specification of subids otherwise will use all subjects in raw file
    back_df = pd.read_csv(os.path.join(input_path, 'memory_2step_background_data.csv'))
    stringify(back_df)
    pickle_path = os.path.join(input_path, 'sub_dicts.pickle')
    with open(pickle_path, 'rb') as handle:
        sub_dict = pickle.load(handle)

    subids = subjects
    assert len(
        set(subids) - set(back_df.subid.unique())) <= 0, "trying to grab participants that don't have memory data!"
    clean_back_df = pd.DataFrame()
    for sub in subids:
        if sub not in back_df.subid.unique():
            print(f'sub memory not found {sub}')
        sub_df = back_df[back_df['subid'] == sub]
        clean_back_df = pd.concat([clean_back_df, sub_df])
    clean_back_df = clean_back_df.reset_index(drop=True)
    if exp_num == 2:
        clean_back_df['accuracy'] = 0

        olds_correct = np.where(((clean_back_df['response'] == 0.0) | (clean_back_df['response'] == 1.0)) & (
                clean_back_df['lure_or_no'] == 'target'))[0]
        news_correct = np.where(((clean_back_df['response'] == 2.0) | (clean_back_df['response'] == 3.0)) & (
                clean_back_df['lure_or_no'] != 'target'))[0]
        clean_back_df.loc[olds_correct, 'accuracy'] = 1
        clean_back_df.loc[news_correct, 'accuracy'] = 1
        clean_back_df['background_stim'] = clean_back_df['backgroundImg'].apply(lambda x: os.path.basename(x)[:-4])
        clean_back_df['left_stim'] = clean_back_df['left_obj'].apply(lambda x: os.path.basename(x)[:-4])
        clean_back_df['right_stim'] = clean_back_df['right_obj'].apply(lambda x: os.path.basename(x)[:-4])
        clean_back_df['inside_or_outside'] = 'outside'
        inside_indices = np.where(clean_back_df['backgroundImg'].str.extract('(forest|desert)').isna())[0]
        clean_back_df.loc[inside_indices, 'inside_or_outside'] = 'inside'
        clean_back_df['subject_high_arm'] = clean_back_df['subid'].apply(lambda x: sub_dict[x]['high_arm'])
        clean_back_df['is_high_arm'] = clean_back_df['inside_or_outside'] == clean_back_df['subject_high_arm']

        mem_df = clean_back_df.groupby(['subid', 'lure_or_no', 'is_high_arm'])['accuracy'].mean().reset_index()
        dprime_df = pd.DataFrame()
        num_target_trials = 8  # number of target trials per low and high arm
        num_mismatch_trials = 6  # number of mismatch trials per low and high arm
        target_adjust = {0: 0.5 / num_target_trials, 1: ((num_target_trials - 0.5) / num_target_trials)}
        mismatch_adjust = {0: 0.5 / num_mismatch_trials, 1: ((num_mismatch_trials - 0.5) / num_mismatch_trials)}
        mismatch_indices = np.where(mem_df['lure_or_no'] == 'lure_but_seen')[0]
        non_mismatch_indices = np.where(mem_df['lure_or_no'] != 'lure_but_seen')[0]
        mem_df.loc[mismatch_indices, 'accuracy'] = mem_df.loc[mismatch_indices, 'accuracy'].replace(mismatch_adjust)
        mem_df.loc[non_mismatch_indices, 'accuracy'] = mem_df.loc[non_mismatch_indices, 'accuracy'].replace(
            target_adjust)
        for sub in mem_df['subid'].unique():
            sub_df = mem_df[mem_df['subid'] == sub].reset_index()
            high_target_mask = np.where((sub_df['is_high_arm'] == True) & (sub_df['lure_or_no'] == 'target'))[0]
            low_target_mask = np.where((sub_df['is_high_arm'] == False) & (sub_df['lure_or_no'] == 'target'))[0]
            high_lure_mask = np.where((sub_df['is_high_arm'] == True) & (sub_df['lure_or_no'] == 'lure'))[0]
            low_lure_mask = np.where((sub_df['is_high_arm'] == False) & (sub_df['lure_or_no'] == 'lure'))[0]
            high_mismatch_mask = np.where((sub_df['is_high_arm'] == True) & (sub_df['lure_or_no'] == 'lure_but_seen'))[
                0]
            low_mismatch_mask = np.where((sub_df['is_high_arm'] == False) & (sub_df['lure_or_no'] == 'lure_but_seen'))[
                0]
            sub_df['dprime_lt_high'] = \
                stats.norm.ppf(sub_df.loc[high_target_mask, 'accuracy'])[0] - \
                stats.norm.ppf(1 - sub_df.loc[high_lure_mask, 'accuracy'])[0]
            sub_df['dprime_lt_low'] = \
                stats.norm.ppf(sub_df.loc[low_target_mask, 'accuracy'])[0] - \
                stats.norm.ppf(1 - sub_df.loc[low_lure_mask, 'accuracy'])[0]
            sub_df['dprime_lbst_high'] = \
                stats.norm.ppf(sub_df.loc[high_target_mask, 'accuracy'])[0] - \
                stats.norm.ppf(1 - sub_df.loc[high_mismatch_mask, 'accuracy'])[0]
            sub_df['dprime_lbst_low'] = \
                stats.norm.ppf(sub_df.loc[low_target_mask, 'accuracy'])[0] - \
                stats.norm.ppf(1 - sub_df.loc[low_mismatch_mask, 'accuracy'])[0]
            dprime_df = pd.concat([dprime_df, sub_df])
            ds = sub_df.isin([np.inf, -np.inf])
            assert ds.sum().max() == 0, 'data contains inf'
        dprime_df.to_csv(os.path.join(output_path, 'dprime_df.csv'))
        clean_back_df.to_csv(os.path.join(output_path, 'memory_df.csv'))


def make_pca_df(input_path, subjects=None):
    lower_indices = np.tril_indices(10, -1, 10)  # m, k n
    slider_pickle_in = os.path.join(input_path, 'slider_dicts.pickle')
    slider_dict = pickle.load(open(slider_pickle_in, 'rb'))
    sub_data = []
    print(f'number of subjects in PCA is {len(subjects)}')
    for sub in subjects:
        post_minus_pre_state1 = slider_dict[sub][1][1] - slider_dict[sub][1][0]
        sub_data.append(post_minus_pre_state1[lower_indices].flatten())
    X = np.asarray(sub_data)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    pca_df = pd.DataFrame()
    pca_df['pc_1'] = pca.components_[0, :]
    pca_df['pc_2'] = pca.components_[1, :]
    pca_df['pc_3'] = pca.components_[2, :]
    pca_df.to_csv(os.path.join(input_path, 'pca_df.csv'))
    return pca_df


def load_exp_data(data_path):
    # loading slider data
    slider_dicts_path = os.path.join(data_path, 'slider_dicts.pickle')
    assert os.path.exists(slider_dicts_path), f'{slider_dicts_path} does not exist!'
    with open(slider_dicts_path, 'rb') as file:
        slider_dict = pickle.load(file)
    model_mat_path = os.path.join(data_path, 'model_mat_fits.csv')
    assert os.path.exists(model_mat_path), f'{model_mat_path} does not exist!'

    model_mat_fits = pd.read_csv(model_mat_path)
    model_mat_fits = model_mat_fits.drop(columns='Unnamed: 0')
    model_mat_fits = model_mat_fits.rename(
        columns={"state1_coef": "Visual cooccurrence", "l_coef": "Direct item association",
                 "o_coef": "Indirect item association"})
    hvl_mat_path = os.path.join(data_path, 'model_mat_fits_hvl.csv')
    assert os.path.exists(hvl_mat_path), f'{hvl_mat_path} does not exist!'
    hvl_df = pd.read_csv(hvl_mat_path)
    melted_mmf = hvl_df[['subid', 'state1_diff', 'l_diff', 'o_diff']].melt(id_vars='subid', var_name='coef',
                                                                           value_name='value')

    # loading rl model fit data
    w1_map_file = os.path.join(data_path, 'w1_map_df.csv')
    assert os.path.exists(w1_map_file), f'{w1_map_file} does not exist!'
    w1_map_df = pd.read_csv(w1_map_file).drop(columns='Unnamed: 0')

    w4_map_file = os.path.join(data_path, 'w4_map_df.csv')

    # loading and formatting stake data
    stake_file = os.path.join(data_path, 'overall_stake_df.csv')
    assert os.path.exists(stake_file), f'{stake_file} does not exist!'
    stake_df = pd.read_csv(stake_file)
    good_indices = np.where(stake_df['rt_2'] != -1)[0]
    good_stakes = stake_df.iloc[good_indices]
    good_stakes.loc[:, 'rews1'] = good_stakes['rews1'] / 9
    good_stakes.loc[:, 'rews2'] = good_stakes['rews2'] / 9
    good_stakes['rews_together'] = (good_stakes['rews1'] + good_stakes['rews2']) / 2

    grouped_stakes = good_stakes.groupby(['subid'])['points', 'rews_together', 'rt_1', 'rt_2'].mean().reset_index()
    grouped_stakes['Points earned in decision-making task'] = grouped_stakes['points'] - grouped_stakes['rews_together']

    pca_file = os.path.join(data_path, 'pca_df.csv')
    if os.path.exists(pca_file):
        pca_df = pd.read_csv(pca_file)
    else:
        print('pca_df not found')
        dprime_df = ''

    dprime_file = os.path.join(data_path, 'dprime_df.csv')
    if os.path.exists(dprime_file):
        dprime_df = pd.read_csv(dprime_file)
    else:
        print('dprime_df not found')
        dprime_df = ''
    if os.path.exists(w4_map_file):
        w4_map_df = pd.read_csv(w4_map_file).drop(columns='Unnamed: 0')
    else:
        print('w4_map_df not found')
        w4_map_df = ''

    data_dict = {'slider_dict': slider_dict, 'model_mat_fits': model_mat_fits, 'model_mat_hvl': hvl_df,
                 'melted_mmf': melted_mmf, 'w1_map_df': w1_map_df, 'w4_map_df': w4_map_df,
                 'grouped_stakes': grouped_stakes, 'dprime_df': dprime_df, 'pca_df': pca_df}
    return data_dict