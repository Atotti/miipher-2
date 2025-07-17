import pandas as pd
from pathlib import Path

def format_value(value, precision=2):
    """数値を指定の精度でフォーマット"""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)

def generate_latex_table():
    """CSVファイルからLaTeXテーブルを生成"""
    results_dir = Path("/home/ayu/GitHub/open-miipher-2/results")

    # CSVファイルを読み込む
    df_8khz = pd.read_csv(results_dir / "summary_8khz.csv")
    df_degrade = pd.read_csv(results_dir / "summary_degrade.csv")

    # LaTeX文書の開始
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[margin=1in]{geometry}
\usepackage{adjustbox}

\begin{document}

\begin{table}[htbp]
  \centering
  \sisetup{
    table-format=1.2,
    round-mode=places,
    round-precision=2
  }

  \caption{音声復元の種類とDNSMOSスコア比較}

  \setlength{\tabcolsep}{4pt}

  \begin{adjustbox}{max width=\linewidth}
    \begin{tabular}{
      l
      l
      *{5}{S[table-format=1.2]}
    }
      \toprule
      \textbf{音声復元の種類} & \textbf{劣化手法}
        & \textbf{ecapa cos} & \textbf{dnsmos p808}
        & \textbf{dnsmos sig} & \textbf{dnsmos bak}
        & \textbf{dnsmos} \\
      \midrule
"""

    # モデル名のマッピング
    name_mapping = {
        'original': 'original',
        '8khz_degraded': 'degraded',
        'noise_degraded': 'degraded',
        'miipher_1': 'miipher-1',
        'hubert_large_l2': 'hubert\\_large\\_l2',
        'mhubert_l6': 'mhubert\\_l6',
        'wav2vec2_base_l2': 'wav2vec2\\_base\\_l2',
        'wavlm_base_l2': 'wavlm\\_base\\_l2'
    }

    # 表示する行の順序
    row_order = ['original', '8khz_degraded', 'miipher_1', 'hubert_large_l2', 'mhubert_l6', 'wav2vec2_base_l2', 'wavlm_base_l2']
    row_order_degrade = ['original', 'noise_degraded', 'miipher_1', 'hubert_large_l2', 'mhubert_l6', 'wav2vec2_base_l2', 'wavlm_base_l2']

    # 8kHz結果のセクション
    num_rows = len(row_order)
    for i, row_name in enumerate(row_order):
        if row_name in df_8khz['name'].values:
            row = df_8khz[df_8khz['name'] == row_name].iloc[0]

            # 最初の行でmultirowを開始
            if i == 0:
                latex_content += f"      \\multirow{{{num_rows}}}{{*}}{{}}\n"

            # モデル名
            model_name = name_mapping.get(row_name, row_name)
            latex_content += f"      {model_name:<17} "

            # 劣化手法（最初の行のみ）
            if i == 0:
                latex_content += f"& \\multirow{{{num_rows}}}{{*}}{{8kHzに変換}}"
            else:
                latex_content += "&     "

            # 数値データ
            latex_content += f"\n                        & {format_value(row['ecapa_cos_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_p808_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_sig_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_bak_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_ovr_mean'])} \\\\\n"

    latex_content += "      \\midrule\n"

    # Degrade結果のセクション
    num_rows = len(row_order_degrade)
    for i, row_name in enumerate(row_order_degrade):
        if row_name in df_degrade['name'].values:
            row = df_degrade[df_degrade['name'] == row_name].iloc[0]

            # 最初の行でmultirowを開始
            if i == 0:
                latex_content += f"      \\multirow{{{num_rows}}}{{*}}{{}}\n"

            # モデル名
            model_name = name_mapping.get(row_name, row_name)
            latex_content += f"      {model_name:<17} "

            # 劣化手法（最初の行のみ）
            if i == 0:
                latex_content += f"& \\multirow{{{num_rows}}}{{*}}{{残響・背景雑音}}"
            else:
                latex_content += "&     "

            # 数値データ
            latex_content += f"\n                        & {format_value(row['ecapa_cos_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_p808_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_sig_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_bak_mean'])}"
            latex_content += f" & {format_value(row['dnsmos_ovr_mean'])} \\\\\n"

    # LaTeX文書の終了
    latex_content += r"""      \bottomrule
    \end{tabular}
  \end{adjustbox}
\end{table}

\end{document}
"""

    # ファイルに保存
    output_path = results_dir / "results_table.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"LaTeX table generated: {output_path}")

    # 画面にも出力
    print("\n" + "="*80)
    print("Generated LaTeX code:")
    print("="*80)
    print(latex_content)

if __name__ == "__main__":
    generate_latex_table()
