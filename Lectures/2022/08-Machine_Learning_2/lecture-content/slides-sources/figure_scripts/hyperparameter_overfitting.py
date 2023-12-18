import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from config import FIGURES_DIR

rng = np.random.default_rng(0)

distrib = stats.norm(0.5, 0.025)
results = distrib.rvs(size=(3000, 100))

fig, ax = plt.subplots(figsize=(3.5, 2.5))
bins = np.linspace(0.4, 0.63, 20)
alpha = 1.0
df = pd.DataFrame(
    {
        "1 candidate": results[0],
        "20 candidates": results[:20].max(axis=0),
        "100 candidates": results.max(axis=0),
    }
)
sns.kdeplot(data=df, ax=ax, fill=True)
handles = ax.legend_.legendHandles
ax.legend(handles, df.columns, frameon=False, loc="upper left")
ax.set_title("Test-set accuracy\nof best model among...")
ax.set_ylabel("")
ax.set_yticks([])

fig.savefig(FIGURES_DIR / "accuracies.pdf", bbox_inches="tight")
