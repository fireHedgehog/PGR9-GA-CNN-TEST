import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def box_plot_per_generation(path, by_generation=False, num_gen=0):
    df = pd.read_csv(path)
    df.columns = ["Batch", "Generation", "Population", "Loss"]

    # check if need to print by generations
    if by_generation:
        result = []
        for name, group in df.groupby(['Batch']):
            # result.append(np.asarray(group["Loss"].values.tolist()))
            result.append(group)

        generation_data = []
        first_gen = result[num_gen]
        for name, group in first_gen.groupby(['Generation']):
            # result.append(np.asarray(np.asarray(group["Loss"].values.tolist())))
            generation_data.append(np.asarray(group["Loss"].values.tolist()))

        # print(len(generation_data))
        every_ten_gen = []
        for idx, gen in enumerate(generation_data):
            if idx == 0 or (idx + 1) % 10 == 0 or idx == 199:
                every_ten_gen.append(gen)
        box_plot_data = every_ten_gen
    else:
        result = []
        for name, group in df.groupby(['Batch']):
            result.append(np.asarray(group["Loss"].values.tolist()))
        box_plot_data = result

    return box_plot_data


def get_min(box_1):
    return_arr = [np.nan]
    for d in box_1:
        return_arr.append(np.min(d))
    return return_arr


def combine_box_plot(box_1, box_2, box_3):
    fig, axs = plt.subplots(3, sharex=True, figsize=(12, 14))

    axs[0].boxplot(box_1, showfliers=False)
    axs[0].plot(get_min(box_1), '--')
    axs[0].set_title("Line Direction Classification(Loss per generation)")
    axs[0].set_ylabel("Loss Value")

    axs[1].boxplot(box_2, showfliers=False)
    axs[1].plot(get_min(box_2), '--')
    axs[1].set_title("Cross Direction Classification(Loss per generation)")
    axs[1].set_ylabel("Loss Value")

    axs[2].boxplot(box_3, showfliers=False)
    axs[2].plot(get_min(box_3), '--')
    axs[2].set_title("Geometric Shape Classification(Loss per generation)")
    axs[2].set_ylabel("Loss Value")
    axs[2].set_xlabel("Generation")
    plt.xticks(ticks=[i for i in range(1, 21)],
               labels=["1", '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110',
                       '120', '130', '140', '150', '160', '170', '180', '190'
                       ], rotation=45)

    plt.show()


def result_line_chart():
    df_cross = pd.read_csv("./history_data/shapes_experiment_no_2/ga_opt_shapes__history.csv")
    df_cross.columns = ["idx", "Batch", "Loss"]

    plt.plot(df_cross["Loss"].tolist(), 'k')
    plt.ylabel('some numbers')
    plt.show()


def show_statistics():
    df = pd.read_csv("./history_data/lines/Line_genetic_algorithm_opt_history.csv")
    df.columns = ["Batch", "Generation", "Population", "Loss"]
    print(df["Loss"].describe())

    df1 = pd.read_csv("./history_data/cross/cross_genetic_algorithm_opt_history.csv")
    df1.columns = ["Batch", "Generation", "Population", "Loss"]
    print(df1["Loss"].describe())

    df2 = pd.read_csv("./history_data/shapes_experiment_no_2/shapes___genetic_algorithm_opt_history.csv")
    df2.columns = ["Batch", "Generation", "Population", "Loss"]
    print(df2["Loss"].describe())


if __name__ == '__main__':
    # result_line_chart() ### obsolete: experiment result not noticeable
    # data_1 = box_plot_per_generation(
    #     path="./history_data/lines/Line_genetic_algorithm_opt_history.csv",
    #     by_generation=True)
    # data_2 = box_plot_per_generation(
    #     path="./history_data/cross/cross_genetic_algorithm_opt_history.csv",
    #     by_generation=True)
    # data_3 = box_plot_per_generation(
    #     path="./history_data/shapes_experiment_no_2/shapes___genetic_algorithm_opt_history.csv",
    #     by_generation=True,
    #     num_gen=0)
    #
    # combine_box_plot(data_1, data_2, data_3)
    show_statistics()
