# %% [markdown]
# ### Ground state energy
# %%
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

# %%
l = "2.0"
ls = l.replace(".", "")
print(f"l{ls}_mini_gs.csv")
data = pd.read_csv(f"l{ls}_mini_gs.csv", header=0, dtype={"Lambda": int})
# %%
fig, ax = plt.subplots()
data[::2].plot(x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (O)", ax=ax)
data[1::2].plot(
    x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (E)", ax=ax
)
ax.axhline(y=0.0, c="k", linestyle="--")
ax.grid()
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.legend(loc="lower right")
plt.savefig(f"l{ls}_mini_gs.png")
plt.savefig(f"l{ls}_mini_gs.pdf")
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
plt.savefig(f"l{ls}_mini_gs_diff.png")
plt.savefig(f"l{ls}_mini_gs_diff.pdf")

# %%
# for this ground state we know the energy should be zero
fig, ax = plt.subplots()
data[::2].abs().plot(
    x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (O)", logy=True, ax=ax
)
data[1::2].abs().plot(
    x="Lambda", y="Energy0", marker="o", label=rf"$\lambda$={l} (E)", logy=True, ax=ax
)
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.grid()
ax.legend(loc="upper right")
plt.savefig(f"l{ls}_mini_gs_log.png")
plt.savefig(f"l{ls}_mini_gs_log.pdf")
# %%
