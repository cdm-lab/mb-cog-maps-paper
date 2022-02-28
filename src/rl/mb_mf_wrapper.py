"""
Model fitting code for the two stage task using MLE (or MAP based on function specified) and multiprocessing
currently bounds are hard coded in the script as opposed to being specified by parameter file
Input: the subject stakes dataframe loaded in from the csv in data
Output: Pickled dictionary file for the subject for n randomized initial parameter model fits
Example: python mb_mf_fit_wrapper_multicore.py -s data/sub_1333_stake.csv -m MB_MF_rllik_final_mat -n 10 -x True
"""
import os
import sys
import time
import argparse
import scipy.optimize
from src.rl.mb_mf_fit import *
import pickle
from multiprocessing import Pool


def fit(sub_df, model_name, stakes, sophistication, final):
    sub = sub_df.iloc[0].subid
    kappa_equivalent = False

    if model_name == 'MB_MF_rllik_final_mat_arms':
        model_function = MB_MF_rllik_learn_mat_arms
        bounds = [(0, 20), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-20, 20), (-20, 20), (0, 0), (0, 0)]
    if model_name == 'MB_MF_rllik_learn_mat_arms':
        model_function = MB_MF_rllik_learn_mat_arms
        if sophistication == 'etakappa':
            bounds = [(0, 20), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-20, 20), (-20, 20), (0, 1), (0, 1)]
        elif sophistication == 'hardcode_etakappa':
            bounds = [(0, 20), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-20, 20), (-20, 20), (1, 1), (1, 1)]
        elif sophistication == 'eta':
            bounds = [(0, 20), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-20, 20), (-20, 20), (0, 1), (1, 1)]
            kappa_equivalent = True

    if model_name == 'MB_MF_MAP_learn_mat_arms':
        model_function = apply_priors
        if sophistication == 'etakappa':
            bounds = [(0.00001, 20), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999),
                      (0.00001, 0.9999), (0.00001, 0.9999), (-20, 20), (-20, 20), (0.00001, 0.9999), (0.00001, 0.9999)]
        elif sophistication == 'hardcode_etakappa':
            bounds = [(0.00001, 20), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999),
                      (0.00001, 0.9999), (0.00001, 0.9999), (-20, 20), (-20, 20), (1.0-1e-9,  1.0), (1.0-1e-9, 1.0)]
        elif sophistication == 'eta':
            bounds = [(0.00001, 20), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999), (0.00001, 0.9999),
                      (0.00001, 0.9999), (0.00001, 0.9999), (-20, 20), (-20, 20), (0.00001, 0.9999), (0.00001, 0.9999)]
            kappa_equivalent = True

    params = param_init(bounds)

    fit = scipy.optimize.minimize(model_function, params, args=([sub_df, stakes, final, kappa_equivalent]), method='L-BFGS-B',
                                      bounds=bounds)

    return [fit, params]


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Model fitting code for the two stage task using MLE or MAP')
    parser.add_argument('-s', '--sub_path', required=True, help='Source file for subject csv')
    parser.add_argument('-m', '--model_name', required=True, help='specify the model type you would like to run')
    parser.add_argument('-n', '--num_inits', required=True,
                        help='specify how many parameter initializations you would like')
    parser.add_argument('-x', '--stakes', required=True, help='how many w parameters to fit [1,2,4]')
    parser.add_argument('-c', '--num_cores', required=True, help='How many cores would you like to run in parallel')
    parser.add_argument('-t', '--sophistication', required=False,
                        help='whether or not people make inferences about the other decision given the one they picked',
                        default=None)
    parser.add_argument('-f', '--final', required=False,
                        help='whether you would like people to start off with final trans_mat', action="store_true",
                        default=False)
    parser.add_argument('-o', '--output_path', required=True, help='point to where you want data saved')
    args = parser.parse_args()
    sub_path = args.sub_path  # Path to the subject data
    if not os.path.exists(sub_path):
        print('%s does not exist!' % sub_path)
        sys.exit(0)
    model_name = args.model_name  # Which model you would like to fit ('current options')
    parameter_initializations = int(args.num_inits)  # how many resets for parameter initialization
    stakes = args.stakes  # whether or not to include stakes
    if stakes not in ['1', '2', '4', '2c']:
        raise ValueError
    num_cores = int(args.num_cores)
    sophistication = args.sophistication

    final = args.final
    output_path = args.output_path

    if (sophistication is not None) & (final is True):
        print('cannot specify sophistication and final_matrix')
        sys.exit(0)
    sub_df = pd.read_csv(sub_path)
    sub = sub_df.iloc[0].subid
    my_args = []
    for i in range(parameter_initializations):
        my_arg_tup = (sub_df, model_name, stakes, sophistication, final)
        my_args.append(my_arg_tup)
    pool = Pool(num_cores)
    results = pool.starmap(fit, my_args)
    if sophistication:
        filename = os.path.join(output_path, f'{sub}_{model_name}_w{stakes}_{sophistication}_transmats_fits.pickle')
    else:
        filename = os.path.join(output_path, f'{sub}_{model_name}_w{stakes}_final_transmats_fits.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle)
    print("--- %s seconds ---" % (np.round(time.time() - start_time,2)))


if __name__ == '__main__':
    main()