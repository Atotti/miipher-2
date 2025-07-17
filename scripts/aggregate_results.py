#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv(csv_path):
    """CSVファイルを読み込む"""
    return pd.read_csv(csv_path)

def calculate_stats(df, name):
    """統計情報を計算"""
    stats = {
        'name': name,
        'ecapa_cos_mean': df['ECAPA_cos'].mean(),
        'ecapa_cos_std': df['ECAPA_cos'].std(),
        'count': len(df)
    }

    # DNSMOSの4つのスコアを処理
    if 'DNSMOS_p808' in df.columns:
        stats.update({
            'dnsmos_p808_mean': df['DNSMOS_p808'].mean(),
            'dnsmos_p808_std': df['DNSMOS_p808'].std(),
            'dnsmos_sig_mean': df['DNSMOS_sig'].mean(),
            'dnsmos_sig_std': df['DNSMOS_sig'].std(),
            'dnsmos_bak_mean': df['DNSMOS_bak'].mean(),
            'dnsmos_bak_std': df['DNSMOS_bak'].std(),
            'dnsmos_ovr_mean': df['DNSMOS_ovr'].mean(),
            'dnsmos_ovr_std': df['DNSMOS_ovr'].std(),
        })
    # 旧DNSMOSv2フォーマットとの互換性
    elif 'DNSMOSv2' in df.columns:
        stats.update({
            'dnsmos_mean': df['DNSMOSv2'].mean(),
            'dnsmos_std': df['DNSMOSv2'].std(),
        })

    return stats

def main():
    # CSVファイルパス
    results_dir = Path("/home/ayu/GitHub/open-miipher-2/results")

    # モデルのサブディレクトリ
    model_dirs = ['hubert_large_l2', 'mhubert_l6', 'miipher_1', 'wav2vec2_base_l2', 'wavlm_base_l2']

    # 元の音声の品質（samples.csv）
    results_original = []
    original_csv_path = results_dir / 'samples.csv'
    if original_csv_path.exists():
        df = load_csv(original_csv_path)
        results_original.append(calculate_stats(df, 'original'))

    # 8kHz劣化後の品質（samples_8khz_16khz.csv）
    results_8khz_baseline = []
    baseline_8khz_csv_path = results_dir / 'samples_8khz_16khz.csv'
    if baseline_8khz_csv_path.exists():
        df = load_csv(baseline_8khz_csv_path)
        results_8khz_baseline.append(calculate_stats(df, '8khz_degraded'))

    # ノイズ劣化後の品質（degrade_samples.csv）
    results_degrade_baseline = []
    baseline_degrade_csv_path = results_dir / 'degrade_samples.csv'
    if baseline_degrade_csv_path.exists():
        df = load_csv(baseline_degrade_csv_path)
        results_degrade_baseline.append(calculate_stats(df, 'noise_degraded'))

    # 8kHzの結果
    results_8khz = []
    for model_dir in model_dirs:
        csv_path = results_dir / model_dir / 'samples_8khz_16khz.csv'
        if csv_path.exists():
            df = load_csv(csv_path)
            results_8khz.append(calculate_stats(df, model_dir))

    # degradeの結果
    results_degrade = []
    for model_dir in model_dirs:
        csv_path = results_dir / model_dir / 'degrade_samples.csv'
        if csv_path.exists():
            df = load_csv(csv_path)
            results_degrade.append(calculate_stats(df, model_dir))

    # 結果を表示
    print("## 8kHz Results (samples_8khz_16khz.csv)")
    # DNSMOSとUTMOSの両方に対応した表示
    first_result = results_8khz[0] if results_8khz else {}
    if 'dnsmos_ovr_mean' in first_result:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOS Overall (mean±std) | DNSMOS SIG | DNSMOS BAK | Samples |")
        print("|--------|---------------------|---------------------------|------------|------------|---------|")
        # 元の音声品質を最初に表示
        if results_original:
            r = results_original[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
        # 8kHz劣化後の品質を表示
        if results_8khz_baseline:
            r = results_8khz_baseline[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
        for r in results_8khz:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
    else:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOSv2 (mean±std) | Samples |")
        print("|--------|---------------------|---------------------|---------|")
        if results_original:
            r = results_original[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")
        if results_8khz_baseline:
            r = results_8khz_baseline[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")
        for r in results_8khz:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")

    print("\n## Degrade Results (degrade_samples.csv)")
    # DNSMOSとUTMOSの両方に対応した表示
    first_degrade = results_degrade[0] if results_degrade else {}
    if 'dnsmos_ovr_mean' in first_degrade:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOS Overall (mean±std) | DNSMOS SIG | DNSMOS BAK | Samples |")
        print("|--------|---------------------|---------------------------|------------|------------|---------|")
        # 元の音声品質を最初に表示
        if results_original:
            r = results_original[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
        # ノイズ劣化後の品質を表示
        if results_degrade_baseline:
            r = results_degrade_baseline[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
        for r in results_degrade:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['dnsmos_sig_mean']:.4f} | {r['dnsmos_bak_mean']:.4f} | {r['count']} |")
    else:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOSv2 (mean±std) | Samples |")
        print("|--------|---------------------|---------------------|---------|")
        if results_original:
            r = results_original[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")
        if results_degrade_baseline:
            r = results_degrade_baseline[0]
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")
        for r in results_degrade:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")

    # 改善率の計算（miipher_1をベースラインとして使用）
    print("\n## Improvement Rates")

    # miipher_1をベースラインとして見つける
    baseline_8khz = None
    baseline_degrade = None

    for r in results_8khz:
        if r['name'] == 'miipher_1':
            baseline_8khz = r
            break

    for r in results_degrade:
        if r['name'] == 'miipher_1':
            baseline_degrade = r
            break

    print("\n### Relative to miipher_1")
    print("\n#### 8kHz")
    if baseline_8khz:
        for method in results_8khz:
            if method['name'] != 'miipher_1':
                ecapa_improve = ((method['ecapa_cos_mean'] - baseline_8khz['ecapa_cos_mean']) / baseline_8khz['ecapa_cos_mean']) * 100

                if 'dnsmos_ovr_mean' in method:
                    dnsmos_improve = ((method['dnsmos_ovr_mean'] - baseline_8khz['dnsmos_ovr_mean']) / baseline_8khz['dnsmos_ovr_mean']) * 100
                    print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOS Overall {dnsmos_improve:+.2f}%")
                elif 'dnsmos_mean' in method:
                    dnsmos_improve = ((method['dnsmos_mean'] - baseline_8khz['dnsmos_mean']) / baseline_8khz['dnsmos_mean']) * 100
                    print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOSv2 {dnsmos_improve:+.2f}%")

    print("\n#### Degrade")
    if baseline_degrade:
        for method in results_degrade:
            if method['name'] != 'miipher_1':
                ecapa_improve = ((method['ecapa_cos_mean'] - baseline_degrade['ecapa_cos_mean']) / baseline_degrade['ecapa_cos_mean']) * 100

                if 'dnsmos_ovr_mean' in method:
                    dnsmos_improve = ((method['dnsmos_ovr_mean'] - baseline_degrade['dnsmos_ovr_mean']) / baseline_degrade['dnsmos_ovr_mean']) * 100
                    print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOS Overall {dnsmos_improve:+.2f}%")
                elif 'dnsmos_mean' in method:
                    dnsmos_improve = ((method['dnsmos_mean'] - baseline_degrade['dnsmos_mean']) / baseline_degrade['dnsmos_mean']) * 100
                    print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOSv2 {dnsmos_improve:+.2f}%")

    # データフレームとして保存
    # 元の音声品質と劣化後の品質を各結果に含める
    combined_8khz = []
    combined_degrade = []

    if results_original:
        combined_8khz.append(results_original[0])
        combined_degrade.append(results_original[0])

    if results_8khz_baseline:
        combined_8khz.append(results_8khz_baseline[0])

    if results_degrade_baseline:
        combined_degrade.append(results_degrade_baseline[0])

    combined_8khz.extend(results_8khz)
    combined_degrade.extend(results_degrade)

    df_8khz = pd.DataFrame(combined_8khz)
    df_degrade = pd.DataFrame(combined_degrade)

    df_8khz.to_csv(results_dir / 'summary_8khz.csv', index=False)
    df_degrade.to_csv(results_dir / 'summary_degrade.csv', index=False)
    print("\n## Summary files saved")
    print(f"- {results_dir / 'summary_8khz.csv'}")
    print(f"- {results_dir / 'summary_degrade.csv'}")

if __name__ == "__main__":
    main()
