#!/usr/bin/env python3
"""
CSVファイルをParquet形式に変換してデータ読み込みを高速化するスクリプト

主な機能:
- メモリ効率的なチャンク処理
- データ型最適化（float64→float32, int64→int32）
- 変換前後のデータ整合性チェック
- 読み込み速度ベンチマーク
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """データ型を最適化してメモリ使用量を削減"""
    print("データ型を最適化中...")
    original_memory = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object":
            # 数値型の最適化
            if "int" in str(col_type):
                # 整数型の最適化
                if (
                    df[col].min() >= np.iinfo(np.int32).min
                    and df[col].max() <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    df[col].min() >= np.iinfo(np.int16).min
                    and df[col].max() <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
            elif "float" in str(col_type):
                # 浮動小数点型の最適化
                df[col] = pd.to_numeric(df[col], downcast="float")

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(
        f"メモリ使用量: {original_memory:.2f}MB → {optimized_memory:.2f}MB "
        f"({((original_memory - optimized_memory) / original_memory * 100):.1f}% 削減)"
    )

    return df


def convert_csv_to_parquet(
    csv_path: Path, parquet_path: Path, chunksize: int = 100000
) -> dict[str, Any]:
    """CSVファイルをParquet形式に変換"""
    print(f"\n=== {csv_path.name} → {parquet_path.name} ===")

    start_time = time.time()
    csv_size = csv_path.stat().st_size / 1024**2  # MB

    # チャンク単位でCSVを読み込み、Parquetに書き込み
    print(f"CSV読み込み開始 (ファイルサイズ: {csv_size:.1f}MB)")

    first_chunk = True
    total_rows = 0
    parquet_writer = None

    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            print(f"チャンク {chunk_idx + 1} 処理中... (行数: {len(chunk)})")

            # データ型最適化
            chunk = optimize_dtypes(chunk)
            total_rows += len(chunk)

            # Parquetファイルに書き込み
            table = pa.Table.from_pandas(chunk)

            if first_chunk:
                parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
                first_chunk = False

            parquet_writer.write_table(table)

    finally:
        if parquet_writer:
            parquet_writer.close()

    conversion_time = time.time() - start_time
    parquet_size = parquet_path.stat().st_size / 1024**2  # MB

    print("変換完了!")
    print(f"  処理時間: {conversion_time:.1f}秒")
    print(f"  総行数: {total_rows:,}")
    print(
        f"  ファイルサイズ: {csv_size:.1f}MB → {parquet_size:.1f}MB "
        f"({((csv_size - parquet_size) / csv_size * 100):.1f}% 削減)"
    )

    return {
        "original_size_mb": csv_size,
        "converted_size_mb": parquet_size,
        "conversion_time_sec": conversion_time,
        "total_rows": total_rows,
        "compression_ratio": csv_size / parquet_size,
    }


def verify_data_integrity(
    csv_path: Path, parquet_path: Path, sample_size: int = 10000
) -> bool:
    """データ整合性をチェック"""
    print("\n=== データ整合性チェック ===")

    # サンプルデータで比較
    print("CSVサンプル読み込み中...")
    csv_sample = pd.read_csv(csv_path, nrows=sample_size)

    print("Parquetサンプル読み込み中...")
    parquet_sample = pd.read_parquet(parquet_path).head(sample_size)

    # 基本チェック
    print("基本情報比較:")
    print(f"  行数: CSV={len(csv_sample)}, Parquet={len(parquet_sample)}")
    print(
        f"  列数: CSV={len(csv_sample.columns)}, Parquet={len(parquet_sample.columns)}"
    )

    # 列名チェック
    if not csv_sample.columns.equals(parquet_sample.columns):
        print("❌ 列名が一致しません")
        return False

    # 数値データの整合性チェック（型変換による精度損失を考慮）
    numeric_cols = csv_sample.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # 最初の5列をチェック
        csv_values = csv_sample[col].fillna(0)
        parquet_values = parquet_sample[col].fillna(0)

        # 相対誤差チェック（型変換による小さな誤差は許容）
        max_error = np.max(
            np.abs(csv_values - parquet_values) / (np.abs(csv_values) + 1e-10)
        )
        if max_error > 1e-6:
            print(f"❌ 列 '{col}' で大きな誤差を検出: {max_error}")
            return False

    print("✅ データ整合性チェック完了")
    return True


def benchmark_loading_speed(
    csv_path: Path, parquet_path: Path, iterations: int = 3
) -> dict[str, float]:
    """読み込み速度をベンチマーク"""
    print("\n=== 読み込み速度ベンチマーク ===")

    csv_times = []
    parquet_times = []

    print("CSV読み込み速度測定...")
    for i in range(iterations):
        start_time = time.time()
        _ = pd.read_csv(csv_path)
        csv_time = time.time() - start_time
        csv_times.append(csv_time)
        print(f"  試行 {i + 1}: {csv_time:.2f}秒")

    print("Parquet読み込み速度測定...")
    for i in range(iterations):
        start_time = time.time()
        _ = pd.read_parquet(parquet_path)
        parquet_time = time.time() - start_time
        parquet_times.append(parquet_time)
        print(f"  試行 {i + 1}: {parquet_time:.2f}秒")

    avg_csv_time = np.mean(csv_times)
    avg_parquet_time = np.mean(parquet_times)
    speedup = avg_csv_time / avg_parquet_time

    print("\n結果:")
    print(f"  CSV平均読み込み時間: {avg_csv_time:.2f}秒")
    print(f"  Parquet平均読み込み時間: {avg_parquet_time:.2f}秒")
    print(f"  速度向上: {speedup:.1f}倍")

    return {
        "csv_avg_time": avg_csv_time,
        "parquet_avg_time": avg_parquet_time,
        "speedup_factor": speedup,
    }


def main():
    """メイン処理"""
    data_dir = Path("data")

    # 変換対象ファイル
    files_to_convert = [
        ("train.csv", "train.parquet"),
        ("test.csv", "test.parquet"),
    ]

    results = {}

    print("🚀 CSV → Parquet変換開始")
    print("=" * 50)

    for csv_name, parquet_name in files_to_convert:
        csv_path = data_dir / csv_name
        parquet_path = data_dir / parquet_name

        if not csv_path.exists():
            print(f"⚠️  {csv_name} が見つかりません。スキップします。")
            continue

        if parquet_path.exists():
            print(f"⚠️  {parquet_name} は既に存在します。上書きします。")

        # 変換実行
        conversion_result = convert_csv_to_parquet(csv_path, parquet_path)

        # データ整合性チェック
        integrity_ok = verify_data_integrity(csv_path, parquet_path)

        # 読み込み速度ベンチマーク（小さいファイルのみ）
        if csv_path.stat().st_size < 500 * 1024 * 1024:  # 500MB未満
            benchmark_result = benchmark_loading_speed(csv_path, parquet_path)
            conversion_result.update(benchmark_result)

        conversion_result["integrity_check"] = integrity_ok
        results[csv_name] = conversion_result

    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎉 変換完了サマリー")
    print("=" * 50)

    total_original_size = 0
    total_converted_size = 0

    for filename, result in results.items():
        print(f"\n📄 {filename}:")
        print(
            f"  ファイルサイズ削減: {result['original_size_mb']:.1f}MB → "
            f"{result['converted_size_mb']:.1f}MB"
        )
        print(f"  圧縮率: {result['compression_ratio']:.1f}倍")
        print(f"  変換時間: {result['conversion_time_sec']:.1f}秒")
        print(f"  データ整合性: {'✅ OK' if result['integrity_check'] else '❌ NG'}")

        if "speedup_factor" in result:
            print(f"  読み込み高速化: {result['speedup_factor']:.1f}倍")

        total_original_size += result["original_size_mb"]
        total_converted_size += result["converted_size_mb"]

    print("\n📊 全体サマリー:")
    print(
        f"  総ファイルサイズ: {total_original_size:.1f}MB → "
        f"{total_converted_size:.1f}MB"
    )
    print(f"  総圧縮率: {total_original_size / total_converted_size:.1f}倍")
    print(f"  節約容量: {total_original_size - total_converted_size:.1f}MB")

    print("\n✨ 変換処理が完了しました！")


if __name__ == "__main__":
    main()
