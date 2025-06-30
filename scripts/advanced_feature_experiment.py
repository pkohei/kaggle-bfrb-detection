#!/usr/bin/env python3
"""
高度統計特徴量実験スクリプト（実験1B）

Issue #4の実験1B: 統計的特徴量拡張
- AdvancedFeatureExtractorによる高度統計特徴量抽出
- GroupKFoldによるデータリーク防止評価
- 目標: 71.86% → 75%以上の精度向上
"""

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

# ローカルmoduleのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ローカルimport
from bfrb.features_advanced import AdvancedFeatureExtractor  # noqa: E402
from bfrb.models import LightGBMModel  # noqa: E402


class AdvancedFeatureExperiment:
    """高度統計特徴量実験クラス"""

    def __init__(
        self, data_dir: Path = Path("data"), results_dir: Path = Path("results")
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)

        # 実験設定
        self.sampling_rate = 20.0  # Hz
        self.n_fft_features = 5
        self.max_correlation_pairs = 20
        self.cv_folds = 5

        # 結果記録
        self.experiment_results = {
            "experiment_id": "1B_advanced_statistical_features",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "config": {
                "sampling_rate": self.sampling_rate,
                "n_fft_features": self.n_fft_features,
                "max_correlation_pairs": self.max_correlation_pairs,
                "cv_folds": self.cv_folds,
            },
            "results": {},
        }

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parquetデータを読み込み"""
        print("データを読み込み中...")

        train_path = self.data_dir / "train.parquet"
        test_path = self.data_dir / "test.parquet"

        if not train_path.exists():
            # Parquetファイルがない場合はCSVから読み込み
            print("ParquetファイルがないためCSVから読み込みます...")
            train_df = pd.read_csv(self.data_dir / "train.csv")
            test_df = pd.read_csv(self.data_dir / "test.csv")

            # メモリ効率化のためデータ型最適化
            for col in train_df.columns:
                if col not in ["sequence_id", "subject", "sequence_type"]:
                    if train_df[col].dtype == "float64":
                        train_df[col] = train_df[col].astype("float32")
                    elif train_df[col].dtype == "int64":
                        train_df[col] = train_df[col].astype("int32")

            for col in test_df.columns:
                if col not in ["sequence_id", "subject", "sequence_type"]:
                    if test_df[col].dtype == "float64":
                        test_df[col] = test_df[col].astype("float32")
                    elif test_df[col].dtype == "int64":
                        test_df[col] = test_df[col].astype("int32")

        else:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)

        print(f"訓練データ: {train_df.shape}")
        print(f"テストデータ: {test_df.shape}")
        print(f"ユニークsequence数 (train): {train_df['sequence_id'].nunique()}")

        return train_df, test_df

    def subsample_for_experiment(
        self, df: pd.DataFrame, n_sequences_per_class: int = 20
    ) -> pd.DataFrame:
        """実験用データサブサンプリング（各クラスから代表的なシーケンスを選択）"""
        print(f"実験用サブサンプリング: 各クラスから{n_sequences_per_class}シーケンス")

        sampled_sequences = []

        # 各クラス（behavior_binary）から代表シーケンスを選択
        for behavior_class in df["behavior_binary"].unique():
            class_df = df[df["behavior_binary"] == behavior_class]
            unique_sequences = class_df["sequence_id"].unique()

            # ランダムサンプリング（再現性のためseed固定）
            np.random.seed(42)
            n_to_sample = min(n_sequences_per_class, len(unique_sequences))
            sampled_seq_ids = np.random.choice(
                unique_sequences, size=n_to_sample, replace=False
            )

            sampled_sequences.extend(sampled_seq_ids)

        # サンプリングしたシーケンスのデータを抽出
        sampled_df = df[df["sequence_id"].isin(sampled_sequences)].copy()

        print(
            f"サンプリング後: {len(sampled_sequences)}シーケンス, "
            f"{sampled_df.shape[0]}行"
        )
        print("クラス分布:", sampled_df["behavior_binary"].value_counts().to_dict())

        return sampled_df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度統計特徴量を抽出"""
        print("高度統計特徴量を抽出中...")

        # AdvancedFeatureExtractorインスタンス作成
        extractor = AdvancedFeatureExtractor(
            sampling_rate=self.sampling_rate,
            n_fft_features=self.n_fft_features,
        )

        # 全特徴量を抽出
        features_df = extractor.extract_all_features(df, group_col="sequence_id")

        # 特徴量サマリーを記録
        feature_summary = extractor.get_feature_importance_summary()
        self.experiment_results["feature_summary"] = feature_summary

        print("特徴量サマリー:")
        for feature_type, count in feature_summary.items():
            print(f"  {feature_type}: {count}次元")

        return features_df

    def prepare_training_data(
        self, train_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """学習用データを準備"""
        print("学習用データを準備中...")

        # sequence_idごとの代表ラベルを取得
        target_df = (
            train_df.groupby("sequence_id")["behavior_binary"].first().reset_index()
        )

        # 特徴量とターゲットを結合
        training_data = features_df.merge(target_df, on="sequence_id", how="inner")

        # 特徴量行列とターゲットベクトルを分離
        feature_cols = [col for col in features_df.columns if col != "sequence_id"]
        X = training_data[feature_cols].values
        y = training_data["behavior_binary"].values
        groups = training_data["sequence_id"].values

        print(f"学習データ準備完了: X={X.shape}, y={y.shape}")
        print(f"クラス分布: {np.bincount(y)}")

        # NaN値をチェック・処理
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"警告: {nan_count}個のNaN値を0で置換します")
            X = np.nan_to_num(X, nan=0.0)

        return X, y, groups

    def run_cross_validation(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> dict[str, Any]:
        """GroupKFoldクロスバリデーション実行"""
        print(f"{self.cv_folds}フォルドクロスバリデーション実行中...")

        # GroupKFold設定
        gkf = GroupKFold(n_splits=self.cv_folds)

        # 結果記録用
        cv_scores = []
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"Fold {fold_idx + 1}/{self.cv_folds} 実行中...")

            # 訓練・検証データ分割
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # LightGBMモデル訓練
            model = LightGBMModel(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )

            model.fit(X_train, y_train)

            # 予測と評価
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            cv_scores.append(accuracy)

            print(f"  Fold {fold_idx + 1} 精度: {accuracy:.4f}")

            # 詳細結果を記録
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "accuracy": float(accuracy),
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "train_distribution": np.bincount(y_train).tolist(),
                    "val_distribution": np.bincount(y_val).tolist(),
                }
            )

        # 統計結果
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        print("\n=== クロスバリデーション結果 ===")
        print(f"平均精度: {mean_score:.4f} ± {std_score:.4f}")
        print(f"各フォルド: {[f'{score:.4f}' for score in cv_scores]}")

        return {
            "mean_accuracy": float(mean_score),
            "std_accuracy": float(std_score),
            "fold_scores": [float(score) for score in cv_scores],
            "fold_details": fold_results,
        }

    def save_results(self, cv_results: dict[str, Any]) -> None:
        """実験結果を保存"""
        # 結果をまとめ
        self.experiment_results["results"] = cv_results
        self.experiment_results["conclusion"] = {
            "target_achieved": cv_results["mean_accuracy"] >= 0.75,
            "improvement_from_baseline": cv_results["mean_accuracy"] - 0.7186,
            "recommendation": (
                "目標達成"
                if cv_results["mean_accuracy"] >= 0.75
                else "更なる特徴量最適化が必要"
            ),
        }

        # JSONファイルに保存
        results_file = (
            self.results_dir
            / f"advanced_feature_experiment_{self.experiment_results['timestamp']}.json"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False)

        print(f"実験結果を保存: {results_file}")

    def run_experiment(self) -> None:
        """実験1Bを実行"""
        print("=== 実験1B: 高度統計特徴量実験 開始 ===\n")

        try:
            # 1. データ読み込み
            train_df, test_df = self.load_data()

            # 2. 実験用サブサンプリング
            sampled_train = self.subsample_for_experiment(
                train_df, n_sequences_per_class=20
            )

            # 3. 特徴量抽出
            features_df = self.extract_features(sampled_train)

            # 4. 学習用データ準備
            X, y, groups = self.prepare_training_data(sampled_train, features_df)

            # 5. クロスバリデーション実行
            cv_results = self.run_cross_validation(X, y, groups)

            # 6. 結果保存
            self.save_results(cv_results)

            print("\n=== 実験1B 完了 ===")
            status = "✅ 達成" if cv_results["mean_accuracy"] >= 0.75 else "❌ 未達成"
            print(f"目標達成状況: {status}")

        except Exception as e:
            print(f"実験中にエラーが発生: {e}")
            raise


def main():
    """メイン実行関数"""
    experiment = AdvancedFeatureExperiment()
    experiment.run_experiment()


if __name__ == "__main__":
    main()
