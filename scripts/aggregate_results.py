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

    # 8kHzの結果
    results_8khz = []
    for csv_file in ['original.csv', '8khz.csv', '8khz_miipher_1.csv', '8khz_miipher_2.csv']:
        df = load_csv(results_dir / csv_file)
        name = csv_file.replace('8khz', '8kHz').replace('_', ' ').replace('.csv', '')
        if name == '8kHz.csv':
            name = '8kHz baseline'
        results_8khz.append(calculate_stats(df, name))

    # degradeの結果
    results_degrade = []
    for csv_file in ['original.csv', 'degrade.csv', 'degrade_miipher_1.csv', 'degrade_miipher_2.csv']:
        df = load_csv(results_dir / csv_file)
        name = csv_file.replace('_', ' ').replace('.csv', '')
        if name == 'degrade.csv':
            name = 'degrade baseline'
        results_degrade.append(calculate_stats(df, name))

    # 結果を表示
    print("## 8kHz Results")
    # DNSMOSとUTMOSの両方に対応した表示
    first_result = results_8khz[0] if results_8khz else {}
    if 'dnsmos_ovr_mean' in first_result:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOS Overall (mean±std) | Samples |")
        print("|--------|---------------------|---------------------------|---------|")
        for r in results_8khz:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['count']} |")
    else:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOSv2 (mean±std) | Samples |")
        print("|--------|---------------------|---------------------|---------|")
        for r in results_8khz:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")

    print("\n## Degrade Results")
    # DNSMOSとUTMOSの両方に対応した表示
    first_degrade = results_degrade[0] if results_degrade else {}
    if 'dnsmos_ovr_mean' in first_degrade:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOS Overall (mean±std) | Samples |")
        print("|--------|---------------------|---------------------------|---------|")
        for r in results_degrade:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_ovr_mean']:.4f}±{r['dnsmos_ovr_std']:.4f} | {r['count']} |")
    else:
        print("\n| Method | ECAPA-cos (mean±std) | DNSMOSv2 (mean±std) | Samples |")
        print("|--------|---------------------|---------------------|---------|")
        for r in results_degrade:
            print(f"| {r['name']} | {r['ecapa_cos_mean']:.4f}±{r['ecapa_cos_std']:.4f} | {r['dnsmos_mean']:.4f}±{r['dnsmos_std']:.4f} | {r['count']} |")

    # 改善率の計算
    print("\n## Improvement Rates")
    print("\n### 8kHz")
    if len(results_8khz) > 1:
        baseline_8khz = results_8khz[0]
        for i in range(1, len(results_8khz)):
            method = results_8khz[i]
            ecapa_improve = ((method['ecapa_cos_mean'] - baseline_8khz['ecapa_cos_mean']) / baseline_8khz['ecapa_cos_mean']) * 100
            
            if 'dnsmos_ovr_mean' in method:
                dnsmos_improve = ((method['dnsmos_ovr_mean'] - baseline_8khz['dnsmos_ovr_mean']) / baseline_8khz['dnsmos_ovr_mean']) * 100
                print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOS Overall {dnsmos_improve:+.2f}%")
            elif 'dnsmos_mean' in method:
                dnsmos_improve = ((method['dnsmos_mean'] - baseline_8khz['dnsmos_mean']) / baseline_8khz['dnsmos_mean']) * 100
                print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOSv2 {dnsmos_improve:+.2f}%")

    print("\n### Degrade")
    if len(results_degrade) > 1:
        baseline_degrade = results_degrade[0]
        for i in range(1, len(results_degrade)):
            method = results_degrade[i]
            ecapa_improve = ((method['ecapa_cos_mean'] - baseline_degrade['ecapa_cos_mean']) / baseline_degrade['ecapa_cos_mean']) * 100
            
            if 'dnsmos_ovr_mean' in method:
                dnsmos_improve = ((method['dnsmos_ovr_mean'] - baseline_degrade['dnsmos_ovr_mean']) / baseline_degrade['dnsmos_ovr_mean']) * 100
                print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOS Overall {dnsmos_improve:+.2f}%")
            elif 'dnsmos_mean' in method:
                dnsmos_improve = ((method['dnsmos_mean'] - baseline_degrade['dnsmos_mean']) / baseline_degrade['dnsmos_mean']) * 100
                print(f"{method['name']}: ECAPA-cos {ecapa_improve:+.2f}%, DNSMOSv2 {dnsmos_improve:+.2f}%")

    # データフレームとして保存
    df_8khz = pd.DataFrame(results_8khz)
    df_degrade = pd.DataFrame(results_degrade)

    df_8khz.to_csv(results_dir / 'summary_8khz.csv', index=False)
    df_degrade.to_csv(results_dir / 'summary_degrade.csv', index=False)
    print("\n## Summary files saved")
    print(f"- {results_dir / 'summary_8khz.csv'}")
    print(f"- {results_dir / 'summary_degrade.csv'}")

if __name__ == "__main__":
    main()
