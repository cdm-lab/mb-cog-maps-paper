import os
import time
import argparse
from src.rl.mb_mf_fit import *
from src.rl.sarsa_agent import *
import pickle
from multiprocessing import Pool


def simulate_agent(
    high_arm, w_count, params, filename, agent_num, final, coin_flip=None
):
    """
    :param high_arm: which arm is the high-context arm (0 or 1)
    :param w_count: how many different w parameters to instantiate
    :param params: the true parameters for the agent
    :param filename: where the data will be saved out for the agent
    :param agent_num: the number of the agent
    :param final: whether or not to do the final transition matrix or learn it with eta and kappa
    :param coin_flip: whether to perform random action or action according the decision rule
    :return: outputs a .pkl file containing the settings and data.csv containing the simulated data for model fitting
    """
    rewards = (0, 2)  # mean and standard deviation of reward
    N = 256
    stakes = (0.8, 0.2)
    rews, state_and_stake = create_task_lists(N, rewards, stakes, high_arm)
    if int(w_count) == 2:
        params[3] = params[5]
        params[4] = params[6]
    actions, rewards, states_traversed = MBMF_deterministic_1choice_rew_sim(
        params,
        rews,
        state_and_stake[:, 0],
        state_and_stake[:, 1],
        final,
        high_arm,
        w_count,
        coin_flip,
    )

    model_df = (
        pd.DataFrame()
    )  
    model_df["state1"] = states_traversed[:, 0]
    model_df["state2"] = states_traversed[:, 1]
    model_df["stake"] = state_and_stake[:, 1]
    model_df["actions"] = actions
    model_df["points"] = rewards / 9
    model_df["high_arm"] = high_arm
    model_df["rt_1"] = 1
    model_df["rt_2"] = 1
    model_df["stim_left_num"] = 1
    model_df["subid"] = int(agent_num)
    model_df["choice1"] = model_df["actions"] + (model_df["state1"] - 1) * 2
    results = [params, w_count]
    model_df.to_csv(filename + ".csv")
    pickle.dump(results, open(filename + ".pkl", "wb"))
    return results


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Simulation code for the two step task"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="point to where you want agent_dfs saved",
    )
    parser.add_argument(
        "-n",
        "--number_of_agents",
        required=True,
        help="specify how many agents you would like",
        type=int,
    )
    parser.add_argument(
        "-w",
        "--w_count",
        required=True,
        help="whether or not to set the w parameters equal to each other for high and low arm",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--num_cores",
        required=True,
        help="How many cores would you like to run in parallel",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--final",
        required=False,
        help="whether you would like people to start off with final trans_mat",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--param_file",
        required=False,
        help="file containing list of parameter fits from subjects",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--random",
        required=False,
        help="whether you would like to update the coin_flip parameter to True",
        action="store_true",
        default=None,
    )
    args = parser.parse_args()
    output_path = args.output_path  # Path to the subject data
    num_agents = args.number_of_agents  # how many resets for parameter initialization
    w_count = args.w_count  # whether or not to include stakes
    num_cores = args.num_cores
    final = args.final
    coin_flip = args.random

    my_args = []
    results = []
    if final:
        bounds = [
            (0, 2),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 0),
            (0, 0),
        ]
    else:
        bounds = [
            (0, 2),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 1),
            (0, 1),
        ]
    my_args = []
    for i in range(num_agents):
        if args.param_file is not None:
            param_file = pd.read_csv(args.param_file)
            params = param_file.loc[
                i,
                [
                    "beta fit",
                    "alpha fit",
                    "lambda fit",
                    "w low stakes low arm fit",
                    "w high stakes low arm fit",
                    "w low stakes high arm fit",
                    "w high stakes high arm fit",
                    "stickiness fit",
                    "resp stickiness fit",
                    "eta fit",
                    "kappa fit",
                ],
            ].values
            if i % 2 == 0:
                high_arm = 1
            else:
                high_arm = 2
        else:
            params = param_init(bounds)
            if i % 2 == 0:
                high_arm = 1
            else:
                high_arm = 2
        filename = os.path.join(output_path, f"agent_{i}_w{int(w_count)}")
        my_args.append((high_arm, w_count, params, filename, i, coin_flip))

    pool = Pool(num_cores)
    results = pool.starmap(simulate_agent, my_args)


if __name__ == "__main__":
    main()
