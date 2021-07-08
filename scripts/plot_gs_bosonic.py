# %% [markdown]
# ### Ground state energy
# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
l = "1.0"
ls = l.replace(".", "")
data = pd.read_csv(f"l{ls}_gs.csv", header=0, dtype={"Lambda": int, "Energy": float})
# %%
fig, ax = plt.subplots()
data[::2].plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} odd", ax=ax)
data[1::2].plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} even", ax=ax)
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.legend(loc="lower right")
plt.savefig(f"l{ls}_gs.pdf")
# %%
fig, ax = plt.subplots()
data.diff()[::2].abs().plot(
    y="Energy", marker="o", label=rf"$\lambda$={l} odd", logy=True, ax=ax
)
data.diff()[1::2].abs().plot(
    y="Energy", marker="o", label=rf"$\lambda$={l} even", logy=True, ax=ax
)
ax.set_ylabel(r"$E_0^{diff}$", rotation=90)
ax.set_xlabel(r"$\Lambda-3$")
ax.legend(loc="upper right")
plt.savefig(f"l{ls}_gs_diff.pdf")

# %%
