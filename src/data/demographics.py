import pandas as pd
import os
import argparse


def print_demographics_info(sub_info_df):
    """
    :param sub_info_df: a dataframe consisting of the subject demographic info for those that made it past threshold
    :return: prints the demographics info
    """
    clean_sub_info_df = sub_info_df.reset_index()
    # cleaning up the data to get proportion of males and females and age range
    clean_sub_info_df['gender'] = clean_sub_info_df['gender'].astype(str).str[0]
    clean_sub_info_df['gender'] = clean_sub_info_df['gender'].str.lower()
    print(clean_sub_info_df.groupby(['gender'])['task'].count())
    print(f'mean age is {clean_sub_info_df["age"].mean()}\n')
    print(f'max age is {clean_sub_info_df["age"].max()}\n')
    print(f'min age is {clean_sub_info_df["age"].min()}\n')
    print(f'age counts are {clean_sub_info_df.groupby(["age"]).count()}\n')


def main():
    parser = argparse.ArgumentParser(description='Code to generate dataframes of subject fits for the two step task')
    parser.add_argument('-i', '--input_path', required=True,
                        help='point to the csv that contains the demographics information')
    parser.add_argument('-s', '--sublist', required=True,
                        help='file containing list of subjects to include in the dataframe')
    parser.add_argument('-o', '--output_path', required=False,
                        help='file containing list of subjects to include in the dataframe')
    args = parser.parse_args()
    input_path = args.input_path
    assert input_path[-3:] == 'csv', 'Need to provide a csv file for the input!'
    sublist_file = args.sublist
    assert os.path.exists(sublist_file), 'provided filename for sublist does not exist!'
    with open(sublist_file) as f:
        subjects = f.readlines()
    subjects = [sub.strip() for sub in subjects]
    input_df = pd.read_csv(input_path)
    sub_info_df = pd.DataFrame()
    for sub in input_df.workerid.unique():
        if sub in subjects:
            sub_df = input_df[input_df['workerid'] == sub]
            sub_info_df = pd.concat([sub_info_df, sub_df])
    print_demographics_info(sub_info_df)


if __name__ == '__main__':
    main()
