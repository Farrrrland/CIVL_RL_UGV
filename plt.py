import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

plt.rc('font', family='serif', weight='medium') 

def smoothed(interval, window_length, k):
    return scipy.signal.savgol_filter(interval, window_length, k)

if __name__ == "__main__":

    fname = "./logs/rl-demo-path-smoke.log"

    rList = []
    for line in open(fname):
        if line.startswith("Epi"):
            rList.append(float(line.split(' ')[3][:-1]))

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=500)

    ax.plot(range(len(rList)), rList, label = "epi_return", color=sns.color_palette("Purples")[2], lw=2)
    ax.plot(range(len(rList)), smoothed(rList, 50, 5), label = "smoothed", color=sns.color_palette("Reds")[3], lw=2)

    ax.set_xlabel("Episode Num.(#)", size = 28)
    ax.set_ylabel("Episodic Return", size = 28)
    plt.yticks(size = 18)
    plt.xticks(size = 18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"./img/q_learning_path_smoke_epi_return.pdf")