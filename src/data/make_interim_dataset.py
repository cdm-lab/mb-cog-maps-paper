import argparse
import os
import src.utils.dataset_utils as ds_utils


def make_dataset(input_path, output_path, subjects, exp):

    print(f"making stakes df at {output_path}")
    ds_utils.make_overall_stake_df(input_path, output_path, subjects)
    print(f"making slider df and slider_hvl_df at {output_path}")
    slider_dict = ds_utils.make_slider_dict(input_path, output_path, subjects)
    ds_utils.make_slider_df(slider_dict, output_path, hvl=False)
    ds_utils.make_slider_df(slider_dict, output_path, hvl=True)
    if exp == 2:
        print(f"making memory df at {output_path}")
        ds_utils.make_memory_df(input_path, output_path, subjects, exp)


def main():
    parser = argparse.ArgumentParser(
        description="Code to generate working dataset csvs for 2step analyses"
    )
    parser.add_argument(
        "-e",
        "--exp",
        required=False,
        help="whether you want analyses on cohort 1 or cohort 2 (for memory)",
        default=2,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="point to where you are looking for files",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        help="where do you want the output to be saved (only flag if you want it saved)",
        default=None,
    )
    args = parser.parse_args()
    input_path = args.input_path
    assert os.path.exists(input_path), "provided input path does not exist!"
    ds_utils.make_sub_dicts(input_path)
    output_path = args.output_path
    subject_csvs_dir = os.path.join(output_path, "subject_csvs")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(subject_csvs_dir):
        os.mkdir(subject_csvs_dir)
    exp = args.exp
    print("creating clean list of subjects because one was not provided")
    subjects = ds_utils.make_good_subjects_list(input_path, output_path, exp)
    print("making interim dataset!")
    make_dataset(input_path, output_path, subjects, exp)


if __name__ == "__main__":
    main()
