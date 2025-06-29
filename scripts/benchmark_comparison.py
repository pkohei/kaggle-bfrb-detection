#!/usr/bin/env python3
"""
データ読み込み速度と精度向上のベンチマーク比較
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def benchmark_loading_speed():
    """CSV vs Parquet読み込み速度比較"""
    print("📊 データ読み込み速度ベンチマーク")
    print("=" * 40)

    results = {}

    # CSV読み込み
    print("CSV読み込み測定中...")
    start_time = time.time()
    _ = pd.read_csv("data/train.csv")
    csv_time = time.time() - start_time
    csv_size = Path("data/train.csv").stat().st_size / 1024**2  # MB

    # Parquet読み込み
    print("Parquet読み込み測定中...")
    start_time = time.time()
    _ = pd.read_parquet("data/train.parquet")
    parquet_time = time.time() - start_time
    parquet_size = Path("data/train.parquet").stat().st_size / 1024**2  # MB

    results = {
        "csv_load_time": csv_time,
        "parquet_load_time": parquet_time,
        "csv_size_mb": csv_size,
        "parquet_size_mb": parquet_size,
        "speedup_factor": csv_time / parquet_time,
        "compression_ratio": csv_size / parquet_size,
    }

    print(f"CSV読み込み時間: {csv_time:.2f}秒")
    print(f"Parquet読み込み時間: {parquet_time:.2f}秒")
    print(f"速度向上: {results['speedup_factor']:.1f}倍")
    print(f"ファイルサイズ: {csv_size:.1f}MB → {parquet_size:.1f}MB")
    print(f"圧縮率: {results['compression_ratio']:.1f}倍")

    return results


def create_comparison_chart():
    """比較チャートを作成"""
    # 実験結果データ
    experiments = {
        "Quick Baseline\n(50k samples)": {
            "accuracy": 0.7361,
            "samples": 50000,
            "data_format": "CSV",
        },
        "Full Dataset\n(575k samples)": {
            "accuracy": 0.8393,
            "samples": 574945,
            "data_format": "Parquet",
        },
    }

    # チャート作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 精度比較
    names = list(experiments.keys())
    accuracies = [exp["accuracy"] for exp in experiments.values()]
    colors = ["lightblue", "lightgreen"]

    bars1 = ax1.bar(names, accuracies, color=colors)
    ax1.set_ylabel("検証精度")
    ax1.set_title("精度比較")
    ax1.set_ylim(0.7, 0.85)

    # 値を表示
    for bar, acc in zip(bars1, accuracies, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # サンプル数比較
    samples = [exp["samples"] for exp in experiments.values()]
    bars2 = ax2.bar(names, samples, color=colors)
    ax2.set_ylabel("サンプル数")
    ax2.set_title("使用データ量比較")
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # 値を表示
    for bar, sample in zip(bars2, samples, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(samples) * 0.01,
            f"{sample:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/experiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    return experiments


def main():
    """メイン処理"""
    print("🚀 全データセット最適化実験 - 総合ベンチマーク")
    print("=" * 60)

    # 読み込み速度ベンチマーク
    loading_benchmark = benchmark_loading_speed()

    # 比較チャート作成
    print("\n📈 比較チャート作成中...")
    experiment_data = create_comparison_chart()

    # 総合レポート
    print("\n" + "=" * 60)
    print("🎉 実験総合レポート")
    print("=" * 60)

    print("📄 データ最適化効果:")
    print(f"  ファイルサイズ削減: {loading_benchmark['compression_ratio']:.1f}倍")
    print(f"  読み込み速度向上: {loading_benchmark['speedup_factor']:.1f}倍")

    print("\n🎯 精度改善効果:")
    baseline_acc = 0.7361
    full_acc = 0.8393
    improvement = full_acc - baseline_acc
    print(f"  ベースライン精度: {baseline_acc:.4f}")
    print(f"  全データ精度: {full_acc:.4f}")
    print(f"  精度向上: +{improvement:.4f} ({improvement / baseline_acc * 100:+.1f}%)")

    print("\n📊 実験スケール:")
    print(f"  使用データ量: 50,000 → 574,945 サンプル ({574945 / 50000:.1f}倍)")
    print(f"  特徴量数: 50 → 333 ({333 / 50:.1f}倍)")

    # 総合結果を保存
    comprehensive_results = {
        "experiment_name": "full_dataset_optimization_benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_optimization": loading_benchmark,
        "accuracy_results": experiment_data,
        "summary": {
            "baseline_accuracy": baseline_acc,
            "optimized_accuracy": full_acc,
            "accuracy_improvement": improvement,
            "improvement_percentage": improvement / baseline_acc * 100,
            "data_scale_increase": 574945 / 50000,
            "feature_scale_increase": 333 / 50,
        },
    }

    # 結果保存
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "comprehensive_benchmark.json", "w") as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n💾 総合結果保存: {output_dir / 'comprehensive_benchmark.json'}")
    print("✨ ベンチマーク完了！")


if __name__ == "__main__":
    main()
