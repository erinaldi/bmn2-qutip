# %% [markdown]
# ### Ground state energy
# %%
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("../figures/paper.mplstyle")
import fire

# %%
def main(l: float = 0.2):
    """Generate plots in the figures folder for the energy from data files in the data folder.
    Only for the ground state.

    Args:
        l (float, optional): The 't Hooft coupling lambda. Defaults to 0.2.
    """
    # %%
    ls = str(l).replace(".", "")
    filename = f"../data/l{ls}_gs.csv"
    print(filename)
    data = pd.read_csv(filename, header=0, dtype={"Lambda": int})
    # %%
    fig, ax = plt.subplots()
    data[::2].plot(
        x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} (O)", ax=ax
    )
    data[1::2].plot(
        x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} (E)", ax=ax
    )
    ax.set_ylabel(r"$E_0$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    ax.grid()
    ax.legend(loc="lower right")
    plt.savefig(f"../figures/l{ls}_gs.pdf")
    plt.savefig(f"../figures/l{ls}_gs.png")
    # %%
    # only keep differences larger than 10-9
    fig, ax = plt.subplots()
    data.diff()[::2].abs().query("Energy > 1e-9").plot(
        y="Energy", marker="o", label=rf"$\lambda$={l} (O)", logy=True, ax=ax
    )
    data.diff()[1::2].abs().query("Energy > 1e-9").plot(
        y="Energy", marker="o", label=rf"$\lambda$={l} (E)", logy=True, ax=ax
    )
    ax.set_ylabel(r"$E_0^{diff}$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    # change ticks such that lambda-3 -> lambda
    ax.xaxis.set_major_formatter(lambda x, p: int(x + 3))
    ax.grid()
    ax.legend(loc="upper right")
    plt.savefig(f"../figures/l{ls}_gs_diff.pdf")
    plt.savefig(f"../figures/l{ls}_gs_diff.png")


# %%
if __name__ == "__main__":
    fire.Fire(main)
