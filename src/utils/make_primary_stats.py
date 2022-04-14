import argparse
import os

import pandas as pd
import pingouin as pg
import numpy as np

import src.utils.dataset_utils as dataset_utils


def model_mat_fits_ttests(model_mat_fits, model_mat_hvl):
    print(f"including {model_mat_fits.subid.nunique()} subjects")
    print("1 sample ttest from 0 for Visual Co-occurence fit")
    print(pg.ttest(model_mat_fits["Visual cooccurrence"], 0))
    print("")
    print("1 sample ttest from 0 for Direct reward association fit")
    print(pg.ttest(model_mat_fits["Direct item association"], 0))
    print("")
    print("1 sample ttest from 0 for Indirect outcome-linked association fit")
    print(pg.ttest(model_mat_fits["Indirect item association"], 0))
    print("")
    print("1 sample ttest from 0 for state1_diff")
    print(pg.ttest(model_mat_hvl["state1_diff"], 0))
    print("")
    print("1 sample ttest from 0 for l_diff")
    print(pg.ttest(model_mat_hvl["l_diff"], 0))
    print("")
    print("1 sample ttest from 0 for o_diff")
    print(pg.ttest(model_mat_hvl["o_diff"], 0))
    print("")


def model_mat_fits_points_corr(model_mat_fits, grouped_stakes):
    model_mat_points = pd.merge(model_mat_fits, grouped_stakes, on="subid")
    melted_mmp = model_mat_points[
        [
            "subid",
            "Visual cooccurrence",
            "Direct item association",
            "Indirect item association",
            "Points earned in decision-making task",
            "rt_2",
        ]
    ].melt(
        id_vars=["subid", "Points earned in decision-making task", "rt_2"],
        var_name="Grouping Model",
        value_name=r"Model Matrix Fit ($\bar{\beta}$)",
    )
    melted_mmp["model_matrix_fit"] = melted_mmp[r"Model Matrix Fit ($\bar{\beta}$)"]
    print("")
    print(f"including {melted_mmp.subid.nunique()} subjects")
    print(
        "Correlation of Visual Co-occurence β with Points earned in decision-making task"
    )
    print(
        pg.corr(
            melted_mmp[melted_mmp["Grouping Model"] == "Visual cooccurrence"][
                "Points earned in decision-making task"
            ],
            melted_mmp[melted_mmp["Grouping Model"] == "Visual cooccurrence"][
                "model_matrix_fit"
            ],
        )
    )
    print(
        "Correlation of Direct reward association β with Points earned in decision-making task"
    )
    print(
        pg.corr(
            melted_mmp[melted_mmp["Grouping Model"] == "Direct item association"][
                "Points earned in decision-making task"
            ],
            melted_mmp[melted_mmp["Grouping Model"] == "Direct item association"][
                "model_matrix_fit"
            ],
        )
    )
    print("")
    print(
        "Correlation of Indirect outcome-linked association β with Points earned in decision-making task"
    )
    print(
        pg.corr(
            melted_mmp[melted_mmp["Grouping Model"] == "Indirect item association"][
                "Points earned in decision-making task"
            ],
            melted_mmp[melted_mmp["Grouping Model"] == "Indirect item association"][
                "model_matrix_fit"
            ],
        )
    )
    print("")


def model_mat_fits_w_corr(model_mat_fits, w1_map_df):
    melted_1w_map = w1_map_df[["subid", "w low stakes low arm fit"]].rename(
        columns={"w low stakes low arm fit": "w fit"}
    )
    model_mat_1w_map = pd.merge(model_mat_fits, melted_1w_map, on="subid")
    mm1w_melt = model_mat_1w_map.melt(
        id_vars=["subid", "high_arm", "w fit"],
        var_name="Grouping Model",
        value_name=r"Model Matrix Fit ($\bar{\beta}$)",
    )
    mm1w_melt["model_matrix_fit"] = mm1w_melt[r"Model Matrix Fit ($\bar{\beta}$)"]
    print(f"including {mm1w_melt.subid.nunique()} subjects")
    print("Correlation of Visual Co-occurrence β with w fit")
    print(
        pg.corr(
            mm1w_melt[mm1w_melt["Grouping Model"] == "Visual cooccurrence"]["w fit"],
            mm1w_melt[mm1w_melt["Grouping Model"] == "Visual cooccurrence"][
                "model_matrix_fit"
            ],
        )
    )

    print("")
    print("Correlation of Direct reward association β with w fit")
    print(
        pg.corr(
            mm1w_melt[mm1w_melt["Grouping Model"] == "Direct item association"][
                "w fit"
            ],
            mm1w_melt[mm1w_melt["Grouping Model"] == "Direct item association"][
                "model_matrix_fit"
            ],
        )
    )

    print("")
    print("Correlation of Indirect outcome-linked association β with w fit")
    print(
        pg.corr(
            mm1w_melt[mm1w_melt["Grouping Model"] == "Indirect item association"][
                "w fit"
            ],
            mm1w_melt[mm1w_melt["Grouping Model"] == "Indirect item association"][
                "model_matrix_fit"
            ],
        )
    )

    print("")


def memory_dprime_ttest(dprime_df):
    dprime_stat_df = dprime_df.groupby(["subid"]).first().reset_index()
    print(f"including {dprime_df.subid.nunique()} subjects")
    print("paired sample t-test between dprime mismatch for high and low arm")
    print(
        pg.ttest(
            dprime_stat_df["dprime_lbst_high"],
            dprime_stat_df["dprime_lbst_low"],
            paired=True,
        )
    )
    print("")
    print("paired sample t-test between dprime lure for high and low arm")
    print(
        pg.ttest(
            dprime_stat_df["dprime_lt_high"],
            dprime_stat_df["dprime_lt_low"],
            paired=True,
        )
    )
    print("")


def model_based_condition_anova(w4_map_df):
    melted_4w_map = w4_map_df[
        [
            "subid",
            "w low stakes low arm fit",
            "w high stakes low arm fit",
            "w low stakes high arm fit",
            "w high stakes high arm fit",
        ]
    ].melt(id_vars=["subid"], var_name="type of", value_name="w")
    melted_4w_map["arm"] = "high"
    melted_4w_map.loc[
        np.where(melted_4w_map["type of"] == "w low stakes low arm fit")[0], "arm"
    ] = "low"
    melted_4w_map.loc[
        np.where(melted_4w_map["type of"] == "w high stakes low arm fit")[0], "arm"
    ] = "low"

    melted_4w_map["stakes"] = "5"
    melted_4w_map.loc[
        np.where(melted_4w_map["type of"] == "w low stakes low arm fit")[0], "stakes"
    ] = "1"
    melted_4w_map.loc[
        np.where(melted_4w_map["type of"] == "w low stakes high arm fit")[0], "stakes"
    ] = "1"
    aov = pg.rm_anova(
        dv="w",
        within=["arm", "stakes"],
        subject="subid",
        data=melted_4w_map,
        detailed=True,
    )
    # Pretty printing of ANOVA summary
    pg.print_table(aov)
    print(
        f"5x w parameter is {melted_4w_map.loc[np.where(melted_4w_map['stakes'] == '5'), 'w'].mean()}"
    )
    print(
        f"1x w parameter is {melted_4w_map.loc[np.where(melted_4w_map['stakes'] == '1'), 'w'].mean()}"
    )


def model_fitting_descriptive_stats(w1_map_df):
    print("Summary stats for different components of model")
    print(
        w1_map_df[
            [
                "beta fit",
                "alpha fit",
                "lambda fit",
                "w low stakes low arm fit",
                "stickiness fit",
                "resp stickiness fit",
                "eta fit",
                "kappa fit",
                "LL",
            ]
        ].mean()
    )
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Code to generate primary stats for behavioral paper"
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
    model_mat_fits = data["model_mat_fits"]
    model_mat_hvl = data["model_mat_hvl"]
    w1_map_df = data["w1_map_df"]
    w4_map_df = data["w4_map_df"]
    if not isinstance(w4_map_df, str):
        model_based_condition_anova(w4_map_df)
    grouped_stakes = data["grouped_stakes"]
    dprime_df = data["dprime_df"]
    # running stats functions to print to terminal TODO come back and make functions write to stats.md file nicely
    model_mat_fits_ttests(model_mat_fits, model_mat_hvl)
    model_mat_fits_points_corr(model_mat_fits, grouped_stakes)
    model_mat_fits_w_corr(model_mat_fits, w1_map_df)
    model_fitting_descriptive_stats(w1_map_df)
    if exp == 2:
        memory_dprime_ttest(dprime_df)


if __name__ == "__main__":
    main()
