# Ad hoc script to create plot for three different languages in one plot, based on existing data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



OUTPUT_DIR = "output"
FILE_NAME = f"output/results-{lang}-train_test-features_set=False-soundclasses=none-use_present=False-ngrams=3-n_runs=1.csv"

dfs = []
for lang in ["latin", "portuguese", "estonian"]:
    df_lang = pd.read_csv(FILENAME, index_col=0)
    df_lang["language"] = lang
    df_lang = df_lang[df_lang["model"]=="ART1"]
    dfs.append(df_lang)
df_total = pd.concat(dfs).reset_index(drop=True)
print(df_total)

df_melt_scores = pd.melt(df_total, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model", "language"], value_vars=[
                                 "ARI"], var_name="metric", value_name="score")
print(df_melt_scores)
ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                     y="score", hue="language", style="mode")
ax_scores.set_ylim(top=1)
plt.savefig(os.path.join(
    OUTPUT_DIR, f"scores-plot-alllangs.pdf"))
plt.clf()