#!/usr/bin/env python3
"""
全データセットを活用した機械学習モデルの訓練スクリプト

主な機能:
- Parquetファイルからの高速データ読み込み
- メモリ効率的な大規模データ処理
- 複数アルゴリズムでの学習と比較
- 層化サンプリングによる適切な検証セット分割
- 詳細な性能評価とレポート生成
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# プロジェクトのソースコードをインポート
sys.path.append("src")


def load_data(use_parquet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """データを読み込み"""
    data_dir = Path("data")

    if use_parquet:
        train_path = data_dir / "train.parquet"
        test_path = data_dir / "test.parquet"

        if not train_path.exists():
            raise FileNotFoundError(f"Parquetファイルが見つかりません: {train_path}")

        print("Parquetファイルから読み込み中...")
        start_time = time.time()
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        load_time = time.time() - start_time
        print(f"読み込み完了: {load_time:.2f}秒")

    else:
        print("CSVファイルから読み込み中...")
        start_time = time.time()
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        load_time = time.time() - start_time
        print(f"読み込み完了: {load_time:.2f}秒")

    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")

    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """特徴量を準備"""
    print("\n=== 特徴量準備 ===")

    # ターゲット列を特定
    target_col = None
    for col in ["behavior", "target", "label"]:
        if col in train_df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("ターゲット列が見つかりません")

    print(f"ターゲット列: {target_col}")

    # 数値特徴量を選択
    feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # ID列を除外
    id_cols = [col for col in feature_cols if "id" in col.lower()]
    for col in id_cols:
        if col in feature_cols:
            feature_cols.remove(col)

    print(f"使用特徴量数: {len(feature_cols)}")
    print(f"除外されたID列: {id_cols}")

    # 特徴量とターゲットを分離
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    X_test = test_df[feature_cols].values

    # ターゲット変数の型をチェック・変換
    print(f"ターゲット変数の型: {y.dtype}")
    print(f"ユニーク値: {np.unique(y)}")

    # 文字列の場合は数値に変換
    if y.dtype == "O":  # object型（文字列）の場合
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"ラベルエンコード後: {np.unique(y)}")
        class_mapping = dict(zip(le.classes_, le.transform(le.classes_), strict=False))
        print(f"クラスマッピング: {class_mapping}")

    # 欠損値チェック
    missing_count = np.isnan(X).sum()
    if missing_count > 0:
        print(f"⚠️  欠損値を検出: {missing_count}個")
        print("欠損値を0で埋めます")
        X = np.nan_to_num(X, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"特徴量行列: {X.shape}")
    print(f"ターゲット: {y.shape}")
    print(f"クラス分布: {np.bincount(y)}")

    return X, y, X_test, feature_cols


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """RandomForestモデルを訓練"""
    print("\n=== RandomForest訓練 ===")

    start_time = time.time()

    # クラス重みを計算（不均衡対策）
    class_weights = "balanced"

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print(f"モデル訓練開始... (サンプル数: {len(X_train):,})")
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 予測と評価
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"訓練時間: {train_time:.1f}秒")
    print(f"検証精度: {accuracy:.4f}")

    return model, {
        "algorithm": "RandomForest",
        "accuracy": accuracy,
        "train_time": train_time,
        "n_estimators": 200,
        "max_depth": 20,
    }


def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    """LightGBMモデルを訓練"""
    print("\n=== LightGBM訓練 ===")

    start_time = time.time()

    # クラス重みを計算
    class_weights = "balanced"

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=0,
        class_weight=class_weights,
        random_state=42,
        n_estimators=500,
    )

    print(f"モデル訓練開始... (サンプル数: {len(X_train):,})")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    train_time = time.time() - start_time

    # 予測と評価
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"訓練時間: {train_time:.1f}秒")
    print(f"検証精度: {accuracy:.4f}")
    print(f"最適反復数: {model.best_iteration_}")

    return model, {
        "algorithm": "LightGBM",
        "accuracy": accuracy,
        "train_time": train_time,
        "n_estimators": model.best_iteration_,
        "learning_rate": 0.05,
    }


def evaluate_model(
    model, X_val: np.ndarray, y_val: np.ndarray, class_names: list[str] = None
) -> dict[str, Any]:
    """詳細なモデル評価"""
    y_pred = model.predict(X_val)

    # 基本メトリクス
    accuracy = accuracy_score(y_val, y_pred)

    # クラス別レポート
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_val)))]

    report = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True
    )

    # 混同行列
    cm = confusion_matrix(y_val, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def create_submission(
    model, X_test: np.ndarray, test_df: pd.DataFrame, algorithm_name: str
) -> str:
    """提出ファイルを作成"""
    predictions = model.predict(X_test)

    # 提出データフレームを作成
    submission_df = pd.DataFrame()

    # IDカラムを探す
    id_col = None
    for col in test_df.columns:
        if "id" in col.lower():
            id_col = col
            break

    if id_col:
        submission_df[id_col] = test_df[id_col]
    else:
        submission_df["id"] = range(len(predictions))

    submission_df["behavior"] = predictions

    # ファイル名を生成
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"full_dataset_{algorithm_name.lower()}_{timestamp}.csv"
    filepath = Path("submissions") / filename

    # 提出ファイルを保存
    submission_df.to_csv(filepath, index=False)
    print(f"提出ファイル保存: {filepath}")

    return str(filepath)


def plot_results(results: list[dict[str, Any]], output_dir: Path):
    """結果を可視化"""
    algorithms = [r["algorithm"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    train_times = [r["train_time"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 精度比較
    bars1 = ax1.bar(algorithms, accuracies, color=["skyblue", "lightgreen"])
    ax1.set_ylabel("検証精度")
    ax1.set_title("モデル精度比較")
    ax1.set_ylim(0, 1)

    # 値をバーの上に表示
    for bar, acc in zip(bars1, accuracies, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    # 訓練時間比較
    bars2 = ax2.bar(algorithms, train_times, color=["orange", "lightcoral"])
    ax2.set_ylabel("訓練時間 (秒)")
    ax2.set_title("訓練時間比較")

    # 値をバーの上に表示
    for bar, time_val in zip(bars2, train_times, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(train_times) * 0.01,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "full_dataset_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def save_results(
    results: list[dict[str, Any]], feature_cols: list[str], output_dir: Path
):
    """結果を保存"""
    # メタデータを追加
    summary = {
        "experiment_name": "full_dataset_optimization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_features": len(feature_cols),
        "feature_names": feature_cols,
        "models": results,
    }

    # JSON形式で保存
    with open(output_dir / "full_dataset_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"結果保存: {output_dir / 'full_dataset_results.json'}")


def main():
    """メイン処理"""
    print("🚀 全データセット学習実験開始")
    print("=" * 50)

    # 出力ディレクトリを作成
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    try:
        # データ読み込み
        train_df, test_df = load_data(use_parquet=True)

        # 特徴量準備
        X, y, X_test, feature_cols = prepare_features(train_df, test_df)

        # 訓練・検証分割（層化サンプリング）
        print("\n=== データ分割 ===")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"訓練セット: {X_train.shape[0]:,} サンプル")
        print(f"検証セット: {X_val.shape[0]:,} サンプル")
        print(f"テストセット: {X_test.shape[0]:,} サンプル")

        # クラス分布を確認
        print("\n訓練セットのクラス分布:")
        for class_id, count in enumerate(np.bincount(y_train)):
            print(f"  クラス {class_id}: {count:,} ({count / len(y_train) * 100:.1f}%)")

        # モデル訓練と評価
        results = []

        # RandomForest
        rf_model, rf_result = train_random_forest(X_train, y_train, X_val, y_val)
        rf_eval = evaluate_model(rf_model, X_val, y_val)
        rf_result.update(rf_eval)
        results.append(rf_result)

        # 提出ファイル作成
        rf_submission = create_submission(rf_model, X_test, test_df, "RandomForest")
        rf_result["submission_file"] = rf_submission

        # LightGBM
        lgb_model, lgb_result = train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_eval = evaluate_model(lgb_model, X_val, y_val)
        lgb_result.update(lgb_eval)
        results.append(lgb_result)

        # 提出ファイル作成
        lgb_submission = create_submission(lgb_model, X_test, test_df, "LightGBM")
        lgb_result["submission_file"] = lgb_submission

        # 結果の可視化と保存
        plot_results(results, output_dir)
        save_results(results, feature_cols, output_dir)

        # サマリー表示
        print("\n" + "=" * 50)
        print("🎉 実験完了サマリー")
        print("=" * 50)

        best_model = max(results, key=lambda x: x["accuracy"])

        for result in results:
            print(f"\n📊 {result['algorithm']}:")
            print(f"  検証精度: {result['accuracy']:.4f}")
            print(f"  訓練時間: {result['train_time']:.1f}秒")
            print(f"  提出ファイル: {result['submission_file']}")

        print(
            f"\n🏆 最高精度: {best_model['algorithm']} ({best_model['accuracy']:.4f})"
        )

        # ベースライン比較
        baseline_accuracy = 0.7361  # 既存のベースライン
        improvement = best_model["accuracy"] - baseline_accuracy
        print(
            f"📈 ベースライン比較: {baseline_accuracy:.4f} → "
            f"{best_model['accuracy']:.4f}"
        )
        print(
            f"   改善: {improvement:+.4f} "
            f"({improvement / baseline_accuracy * 100:+.1f}%)"
        )

        print("\n✨ 全データセット学習実験が完了しました！")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
