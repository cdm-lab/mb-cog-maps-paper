import argparse
import os
import pandas as pd
import src.utils.dataset_utils as ds_utils


def make_final_dataset(input_path, output_path, subjects, exp):
    print(f'making stakes df at {output_path}')
    overall_df = pd.read_csv(os.path.join(input_path, 'overall_stake_df.csv'))
    filter_overall_df = ds_utils.filter_df(overall_df, subjects)
    filter_overall_df.to_csv(os.path.join(output_path, 'overall_stake_df.csv'))
    print(f'making slider df and slider_hvl_df at {output_path}')
    slider_df = pd.read_csv(os.path.join(input_path, 'model_mat_fits.csv'))
    filter_slider_df = ds_utils.filter_df(slider_df, subjects)
    filter_slider_df.to_csv(os.path.join(output_path, 'model_mat_fits.csv'))
    hvl_df = pd.read_csv(os.path.join(input_path, 'model_mat_fits_hvl.csv'))
    filter_hvl_df = ds_utils.filter_df(hvl_df, subjects)
    filter_hvl_df.to_csv(os.path.join(output_path, 'model_mat_fits_hvl.csv'))
    print(f'making new slider_dicts.pickle at {output_path}')
    ds_utils.filter_dict(input_path, output_path, subjects)
    print(f'making w1_map_df at {output_path}')
    w1_map_df = pd.read_csv(os.path.join(input_path, 'w1_map_df.csv'))
    filter_w1_map_df = ds_utils.filter_df(w1_map_df, subjects)
    filter_w1_map_df.to_csv(os.path.join(output_path, 'w1_map_df.csv'))
    if exp == 2:
        print(f'making memory df at {output_path}')
        mem_df = pd.read_csv(os.path.join(input_path, 'memory_df.csv'))
        filter_mem_df = ds_utils.filter_df(mem_df, subjects)
        filter_mem_df.to_csv(os.path.join(output_path, 'memory_df.csv'))
        dprime_df = pd.read_csv(os.path.join(input_path, 'dprime_df.csv'))
        dprime_df = ds_utils.filter_df(dprime_df, subjects)
        dprime_df.to_csv(os.path.join(output_path, 'dprime_df.csv'))
        print(f'making pca_df at {output_path}')
        pca_df = ds_utils.make_pca_df(input_path, output_path, subjects)


def main():
    parser = argparse.ArgumentParser(description='Code to generate working dataset csvs for 2step analyses')
    parser.add_argument('-e', '--exp', required=False,
                        help='whether you want analyses on cohort 1 or cohort 2 (for memory)', default=2, type=int)
    parser.add_argument('-i', '--input_path', required=True, help='point to where you are looking for files')
    parser.add_argument('-o', '--output_path', required=False,
                        help='where do you want the output to be saved (only flag if you want it saved)', default=None)
    args = parser.parse_args()
    input_path = args.input_path
    assert os.path.exists(input_path), 'provided input path does not exist!'

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    exp = args.exp
    print('creating clean list of subjects because one was not provided')
    sublist_file = os.path.join(input_path, f'exp{exp}_good_subjects.txt')
    with open(sublist_file) as f:
        subjects = f.readlines()
    subjects = [sub.strip() for sub in subjects]
    subjects = ds_utils.model_fit_threshold(input_path, output_path, subjects, exp)
    print('making final dataset!')
    make_final_dataset(input_path, output_path, subjects, exp)


if __name__ == '__main__':
    main()
