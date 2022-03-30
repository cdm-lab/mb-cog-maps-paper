import argparse
import os

import pandas as pd
import pingouin as pg

import src.utils.dataset_utils as dataset_utils

# TODO add recovery correlations
# TODO add ek1 parameter correlations
# TODO add stakexarm ANOVA


def w1_points_rt_corr(w1_map_df, grouped_stakes):
    melted_1w_map = w1_map_df[["subid", "w low stakes low arm fit"]].rename(
        columns={"w low stakes low arm fit": "w fit"}
    )
    perfvsw = pd.merge(melted_1w_map, grouped_stakes, on=["subid"])
    pcor = pg.pairwise_corr(
        perfvsw,
        columns=["w fit", "Points earned in decision-making task", "rt_1"],
        method="pearson",
    )
    print(f"including {perfvsw.subid.nunique()} subjects")
    print("Pairwise correlations between w fit, points, and rt")
    print(pcor.round(4))


def dprime_corr(w1_map_df, dprime_df):
    melted_1w_map = w1_map_df[["subid", "w low stakes low arm fit"]].rename(
        columns={"w low stakes low arm fit": "w fit"}
    )
    dprime_stat_df = dprime_df.groupby(["subid"]).first().reset_index()
    dprime_mb = pd.merge(melted_1w_map, dprime_stat_df, on=["subid"])

    print(f"including {dprime_df.subid.nunique()} subjects")
    pcor = pg.pairwise_corr(
        dprime_mb,
        columns=[
            "w fit",
            "dprime_lbst_high",
            "dprime_lbst_low",
            "dprime_lt_high",
            "dprime_lt_low",
        ],
        method="pearson",
    )
    print(
        pcor.round(4).iloc[0:4]
    )  # only printing correlations with w fit because we don't care about dprime correlations with each other


def model_mat_fits_corr(model_mat_fits):
    pcor = pg.pairwise_corr(
        model_mat_fits,
        columns=[
            "Visual cooccurrence",
            "Direct item association",
            "Indirect item association",
        ],
        method="pearson",
    )
    print(pcor.round(4))  #


def main():
    parser = argparse.ArgumentParser(
        description="Code to generate supplementary stats for behavioral paper"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="point to where you are looking for files",
    )
    parser.add_argument(
        "-e",
        "--exp",
        required=False,
        help="specify which experiment version you are running",
        default=2,
        type=int,
    )
    # parser.add_argument('-o', '--output_path', required=True, help='where do you want the figures to be saved')
    args = parser.parse_args()
    input_path = args.input_path
    exp = args.exp
    # output_path = args.output_path
    assert os.path.exists(input_path), "provided input path does not exist!"
    # assert os.path.exists(output_path), 'provided output path does not exist!'
    data = dataset_utils.load_exp_data(input_path)
    w1_map_df = data["w1_map_df"]
    grouped_stakes = data["grouped_stakes"]
    dprime_df = data["dprime_df"]
    model_mat_fits = data["model_mat_fits"]
    # running stats functions to print to terminal
    w1_points_rt_corr(w1_map_df, grouped_stakes)
    model_mat_fits_corr(model_mat_fits)
    if exp == 2:
        dprime_corr(w1_map_df, dprime_df)


if __name__ == "__main__":
    main()
