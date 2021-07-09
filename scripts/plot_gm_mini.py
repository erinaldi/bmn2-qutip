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
def main(l: float = 0.2, J: float = 0.0):
    """Generate plots in the figures folder for the angular momentum difference from J using data in the data folder
    Only for the ground state.

    Args:
        l (float, optional): the 'f Hooft coupling lambda. Defaults to 0.2.
        J (float, optional): the angular momentum of choice J. Defaults to 0.0.
    """
    ls = str(l).replace(".", "")
    Js = str(J).replace(".", "")
    filename = f"../data/l{ls}_mini_gm_J{Js}.csv"
    print(filename)
    data = pd.read_csv(filename, header=0, dtype={"Lambda": int})
    # to subtract the J value and take the absolut values do this
    data_abs = data.apply(lambda x: x - J if x.name != "Lambda" else x).abs()
    # %%
    fig, ax = plt.subplots()
    data_abs[::2].plot(
        x="Lambda",
        y="AngMomentum0",
        marker="o",
        label=fr"$\lambda$={l} (O)",
        logy=True,
        ax=ax,
    )
    data_abs[1::2].plot(
        x="Lambda",
        y="AngMomentum0",
        marker="o",
        label=fr"$\lambda$={l} (E)",
        logy=True,
        ax=ax,
    )
    ax.grid()
    ax.set_ylabel(r"$|\langle \hat{M} \rangle - J|$", rotation=90)
    ax.set_xlabel(r"$\Lambda$")
    plt.savefig(f"../figures/l{ls}_mini_gm_J{Js}.png")
    plt.savefig(f"../figures/l{ls}_mini_gm_J{Js}.pdf")


# %%
if __name__ == "__main__":
    fire.Fire(main)
