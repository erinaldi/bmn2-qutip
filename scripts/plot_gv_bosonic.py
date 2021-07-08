# %% [markdown]
# ### Gauge Violation term
# %%
import pandas as pd
from matplotlib import pyplot as plt

l = "10.0"
ls = l.replace(".", "")
data = pd.read_csv(
    f"l{ls}_gv.csv", header=0, dtype={"Lambda": int, "GaugeViolation": float}
)
# %%
fig, ax = plt.subplots()
data[::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} odd",
    logy=True,
    ax=ax,
)
data[1::2].plot(
    x="Lambda",
    y="GaugeViolation",
    marker="o",
    label=fr"$\lambda$={l} even",
    logy=True,
    ax=ax,
)
ax.set_ylabel(r"$\langle G^2 \rangle$", rotation=90)
ax.set_xlabel(r"$\Lambda$")
plt.savefig(f"l{ls}_gv.pdf")
# %%
