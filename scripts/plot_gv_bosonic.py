# %% [markdown]
# ### Gauge Violation term
# %%
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')

# %% 
l = "2.0"
ls = l.replace(".", "")
print(f"l{ls}_gv.csv")
data = pd.read_csv(
    f"l{ls}_gv.csv", header=0, dtype={"Lambda": int, "GaugeViolation": float}
)
# %%
fig, ax = plt.subplots()
data[::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} (O)",
    logy=True,
    ax=ax,
)
data[1::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} (E)",
    logy=True,
    ax=ax,
)
ax.set_ylabel(r"$\langle G^2 \rangle$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
ax.grid()
plt.savefig(f"l{ls}_gv.pdf")
plt.savefig(f"l{ls}_gv.png")
# %%
