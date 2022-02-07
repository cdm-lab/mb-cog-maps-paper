import os
import pickle
import pandas as pd
import glob
import argparse


def create_fit_df(subjects, column_names, input_path):
    fit_df = pd.DataFrame()
    for sub in subjects:
        sub_file = f"{sub}*.pickle"
        full_path = os.path.join(input_path, sub_file)
        if len(glob.glob(full_path)) == 0:
            print(f'missing {sub}')
            continue
        elif len(glob.glob(full_path)) > 0:
            filename = glob.glob(full_path)[0]
        if os.path.exists(filename):
            data = pickle.load(open(filename, "rb"))
            best_key = 0
            for i in range(len(data)):
                if data[i][0].fun < data[best_key][0].fun:
                    best_key = i
            sub_data = list(zip(data[best_key][1], data[best_key][0].x))
            flat_sub_data = [item for sublist in sub_data for item in sublist]
            sub_fit_df = pd.DataFrame(flat_sub_data).T.rename(columns=column_names)
            sub_fit_df['subid'] = sub
            sub_fit_df['LL'] = data[best_key][0].fun
            fit_df = pd.concat([fit_df, sub_fit_df])
        else:
            print(f'missing {sub}')

    fit_df = fit_df.reset_index().drop(columns='index')
    return fit_df


def main():
    parser = argparse.ArgumentParser(description='Code to generate dataframes of subject fits for the two step task')
    parser.add_argument('-i', '--input_path', required=True, help='point to where you are pulling pickle files from')
    parser.add_argument('-s', '--sublist', required=True,
                        help='file containing list of subjects to include in the dataframe')
    parser.add_argument('-o', '--output_file', required=True,
                        help='where do you want the dataframe saved and what name')
    parser.add_argument('-c', '--colname_list', required=False,
                        help='file containing list of names for dataframe column', default=None)
    args = parser.parse_args()
    input_path = args.input_path
    sublist_file = args.sublist
    output_file = args.output_file
    assert os.path.exists(input_path), 'provided input path for data does not exist!'
    assert os.path.exists(sublist_file), 'provided filename for sublist does not exist!'
    with open(sublist_file) as f:
        subjects = f.readlines()
    subjects = [sub.strip() for sub in subjects]
    colname_list = args.colname_list
    if colname_list is None:
        column_names = {0: 'beta init', 1: 'beta fit',
                        2: 'alpha init', 3: 'alpha fit',
                        4: 'lambda init', 5: 'lambda fit',
                        6: 'w low stakes low arm init', 7: 'w low stakes low arm fit',
                        8: 'w high stakes low arm init', 9: 'w high stakes low arm fit',
                        10: 'w low stakes high arm init', 11: 'w low stakes high arm fit',
                        12: 'w high stakes high arm init', 13: 'w high stakes high arm fit',
                        14: 'stickiness init', 15: 'stickiness fit',
                        16: 'resp stickiness init', 17: 'resp stickiness fit',
                        18: 'eta init', 19: 'eta fit',
                        20: 'kappa init', 21: 'kappa fit', 22: 'LL'}
    else:
        assert os.path.exists(colname_list), 'provided filename for column names does not exist!'
        with open(colname_list) as f:
            col_names = f.readlines()
        col_names = [name.strip() for name in col_names]
        column_names = {i: col_names[i] for i in range(len(col_names))}

    df = create_fit_df(subjects, column_names, input_path)
    df.to_csv(output_file)


if __name__ == '__main__':
    main()
