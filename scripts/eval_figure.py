import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# noise_mode = "degrade"
noise_mode = "PA_E3"


df_v2 = pd.read_csv(f"results/{noise_mode}_miipher_2.csv")
df_v1 = pd.read_csv(f"results/{noise_mode}_miipher.csv")

# データを整形
metrics = ["MCD", "XvecCos", "ECAPACos", "WER", "logF0_RMSE"]

# Miipher v2のデータ
df_v2_restored = df_v2[metrics]
df_v2_restored["Condition"] = "Miipher2"

# Miipher v1のデータ
df_v1_restored = df_v1[metrics]
df_v1_restored["Condition"] = "Miipher"

# Degradedのデータ (どちらのファイルも同じはずなのでv2から使用)
df_degraded = df_v2[[f"Deg_{m}" for m in metrics]]
df_degraded.columns = metrics
df_degraded["Condition"] = "Degraded"

# 3つの条件を結合
df_combined = pd.concat([df_degraded, df_v1_restored, df_v2_restored])

# プロットしやすいようにさらに整形 (Melt)
df_melted = df_combined.melt(id_vars=["Condition"], var_name="Metric", value_name="Value")

# 指標ごとに「高いほど良い」か「低いほど良い」かを定義
lower_is_better = ["MCD", "WER", "logF0_RMSE"]

# 描画
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=df_melted,
    x="Condition",
    y="Value",
    col="Metric",
    kind="box",
    order=["Degraded", "Miipher", "Miipher2"],
    palette={"Degraded": "lightcoral", "Miipher": "skyblue", "Miipher2": "mediumseagreen"},
    height=4.5,
    aspect=0.8,
    sharey=False,
)

# タイトルとラベルを調整
g.fig.suptitle("Performance Comparison: Miipher vs Miipher2", y=1.03, size=18, weight="bold")
g.set_xlabels("")
g.set_ylabels("Metric Value")

for _i, ax in enumerate(g.axes.flat):
    metric_name = ax.get_title().split("= ")[1]
    if metric_name in lower_is_better:
        ax.set_title(f"{metric_name}\n(Lower is better)")
    else:
        ax.set_title(f"{metric_name}\n(Higher is better)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"results/{noise_mode}_miipher_vs_miipher2.png")
