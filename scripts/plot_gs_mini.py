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
    ls = str(l).replace(".", "")
    filename = f"../data/l{ls}_mini_gs.csv"
    print(filename)
    data = pd.read_csv(filename, header=0, dtype={"Lambda": int})
    # %%
    fig, ax = plt.subplots()
    data[::2].plot(
        x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (O)", ax=ax
    )
    data[1::2].plot(
        x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (E)", ax=ax
    )
    ax.axhline(y=0.0, c="k", linestyle="--")
    ax.grid()
    ax.set_ylabel(r"$E_0$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    ax.legend(loc="lower right")
    plt.savefig(f"../figures/l{ls}_mini_gs.png")
    plt.savefig(f"../figures/l{ls}_mini_gs.pdf")
    # %%
    fig, ax = plt.subplots()
    data.diff()[::2].abs().plot(
        y="Energy0", marker="o", label=rf"$\lambda$={l} (O)", logy=True, ax=ax
    )
    data.diff()[1::2].abs().plot(
        y="Energy0", marker="o", label=rf"$\lambda$={l} (E)", logy=True, ax=ax
    )
    ax.set_ylabel(r"$E_0^{diff}$", rotation=90)
    ax.set_xlabel(r"$\Lambda-3$")
    ax.grid()
    ax.legend(loc="upper right")
    plt.savefig(f"../figures/l{ls}_mini_gs_diff.png")
    plt.savefig(f"../figures/l{ls}_mini_gs_diff.pdf")

    # %%
    # for this ground state we know the energy should be zero
    fig, ax = plt.subplots()
    data[::2].abs().plot(
        x="Lambda",
        y="Energy0",
        marker="o",
        label=rf"$\lambda$={l} (O)",
        logy=True,
        ax=ax,
    )
    data[1::2].abs().plot(
        x="Lambda",
        y="Energy0",
        marker="o",
        label=rf"$\lambda$={l} (E)",
        logy=True,
        ax=ax,
    )
    ax.set_ylabel(r"$E_0$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    ax.grid()
    ax.legend(loc="upper right")
    plt.savefig(f"../figures/l{ls}_mini_gs_log.png")
    plt.savefig(f"../figures/l{ls}_mini_gs_log.pdf")


# %%
if __name__ == "__main__":
    fire.Fire(main)
