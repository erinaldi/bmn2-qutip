# %% [markdown]
# ### Ground state energy
# %%
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

# %%
l = "0.2"
ls = l.replace(".", "")
print(f"l{ls}_gs.csv")
data = pd.read_csv(f"l{ls}_gs.csv", header=0, dtype={"Lambda": int, "Energy": float})
# %%
fig, ax = plt.subplots()
data[::2].plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} (O)", ax=ax)
data[1::2].plot(x="Lambda", y="Energy", marker="o", label=rf"$\lambda$={l} (E)", ax=ax)
ax.set_ylabel(r"$E_0$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.grid()
ax.legend(loc="lower right")
plt.savefig(f"l{ls}_gs.pdf")
plt.savefig(f"l{ls}_gs.png")
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
ax.set_xlabel(r"$\Lambda-3$")
ax.grid()
ax.legend(loc="upper right")
plt.savefig(f"l{ls}_gs_diff.pdf")
plt.savefig(f"l{ls}_gs_diff.png")

# %%
