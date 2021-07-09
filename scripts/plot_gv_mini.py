# %% [markdown]
# ### Gauge Violation term
# %%
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("../figures/paper.mplstyle")
import fire

# %%
def main(l: float = 0.2):
    """Generate plots in the figures folder for the gauge constraint violation from data files in the data folder.
    Only for the ground state.

    Args:
        l (float, optional): The 't Hooft coupling lambda. Defaults to 0.2.
    """
    ls = str(l).replace(".", "")
    filename = f"../data/l{ls}_mini_gv.csv"
    print(filename)
    data = pd.read_csv(filename, header=0, dtype={"Lambda": int})
    # %%
    fig, ax = plt.subplots()
    data[::2].plot(
        x="Lambda",
        y="GaugeViolation0",
        marker="o",
        label=fr"$\lambda$={l} (O)",
        logy=True,
        ax=ax,
    )
    data[1::2].plot(
        x="Lambda",
        y="GaugeViolation0",
        marker="o",
        label=fr"$\lambda$={l} (E)",
        logy=True,
        ax=ax,
    )
    ax.grid()
    ax.set_ylabel(r"$\langle G^2 \rangle$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    plt.savefig(f"../figures/l{ls}_mini_gv.png")
    plt.savefig(f"../figures/l{ls}_mini_gv.pdf")


# %%
if __name__ == "__main__":
    fire.Fire(main)
