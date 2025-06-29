#!/usr/bin/env python3
"""
特徴量エンジニアリング実験スクリプト（技術調査結果最適化版）

Issue #4に基づく段階的実験：
- 実験1A: 最適時間窓パラメータ決定
- 実験1B: 統計的特徴量最適化
- 実験2: 周波数領域特徴量導入
- 実験3: 動的特徴量と特徴量選択
- 実験4: 最終最適化
"""

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).parent.parent / "src"))

from bfrb.models import LightGBMModel


class OptimizedFeatureExtractor:
    """技術調査結果に基づく最適化特徴量抽出器"""

    def __init__(self):
        self.sampling_rate = None
        self.feature_names_ = []

    def extract_time_window_features(
        self,
        df: pd.DataFrame,
        window_size_samples: int = 100,
        overlap_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        最適化された時間窓処理による特徴量抽出

        Args:
            df: センサーデータ（時系列順にソート済み）
            window_size_samples: 窓サイズ（サンプル数）
            overlap_ratio: 重複率（0.0-1.0）
        """
        window_size = max(1, window_size_samples)
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        # センサー列を特定（実際のデータ構造に合わせて修正）
        meta_cols = [
            "row_id",
            "sequence_type",
            "sequence_id",
            "sequence_counter",
            "subject",
            "orientation",
            "behavior",
            "behavior_binary",
            "phase",
            "gesture",
        ]
        sensor_cols = [col for col in df.columns if col not in meta_cols]

        features_list = []

        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i : i + window_size]

            # ウィンドウごとの特徴量
            window_features = {}

            # 基本情報
            if "sequence_id" in df.columns:
                window_features["sequence_id"] = window_data["sequence_id"].iloc[0]
            if "behavior_binary" in df.columns:
                # 窓内のbehavior_binaryの最頻値を使用
                behavior_mode = window_data["behavior_binary"].mode()
                window_features["target"] = (
                    behavior_mode.iloc[0] if len(behavior_mode) > 0 else 0
                )

            # 窓のメタ情報
            window_features["window_start"] = i
            window_features["window_size"] = window_size

            # 各センサーの基本統計特徴量
            for col in sensor_cols:
                if col in window_data.columns:
                    values = window_data[col].values

                    # NaN値の処理
                    values = values[~np.isnan(values)]
                    if len(values) == 0:
                        continue

                    # 基本統計量
                    window_features[f"{col}_mean"] = np.mean(values)
                    window_features[f"{col}_std"] = np.std(values)
                    window_features[f"{col}_min"] = np.min(values)
                    window_features[f"{col}_max"] = np.max(values)
                    window_features[f"{col}_median"] = np.median(values)

                    # 追加統計量（技術調査推奨）
                    window_features[f"{col}_range"] = np.max(values) - np.min(values)
                    window_features[f"{col}_var"] = np.var(values)
                    window_features[f"{col}_skewness"] = (
                        stats.skew(values) if len(values) > 1 else 0
                    )
                    window_features[f"{col}_kurtosis"] = (
                        stats.kurtosis(values) if len(values) > 1 else 0
                    )
                    window_features[f"{col}_rms"] = np.sqrt(np.mean(values**2))

                    # パーセンタイル
                    window_features[f"{col}_q25"] = np.percentile(values, 25)
                    window_features[f"{col}_q75"] = np.percentile(values, 75)
                    window_features[f"{col}_iqr"] = np.percentile(
                        values, 75
                    ) - np.percentile(values, 25)

            features_list.append(window_features)

        features_df = pd.DataFrame(features_list)
        self.feature_names_ = [
            col
            for col in features_df.columns
            if col not in ["sequence_id", "target", "window_start", "window_size"]
        ]

        return features_df

    def extract_frequency_features(
        self,
        df: pd.DataFrame,
        window_size_seconds: float = 5.0,
        overlap_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """周波数領域特徴量の抽出"""
        if self.sampling_rate is None:
            self.sampling_rate = 1.0

        window_size = max(1, int(window_size_seconds * self.sampling_rate))
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        sensor_cols = [
            col
            for col in df.columns
            if col not in ["id", "target", "behavior", "timestamp"]
        ]

        features_list = []

        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i : i + window_size]
            window_features = {}

            if "id" in df.columns:
                window_features["id"] = window_data["id"].iloc[0]
            if "target" in df.columns:
                window_features["target"] = (
                    window_data["target"].mode().iloc[0]
                    if len(window_data["target"].mode()) > 0
                    else 0
                )

            for col in sensor_cols:
                if col in window_data.columns:
                    values = window_data[col].values

                    # FFT特徴量
                    fft_values = np.abs(fft(values))[: len(values) // 2]

                    if len(fft_values) > 0:
                        window_features[f"{col}_fft_max"] = np.max(fft_values)
                        window_features[f"{col}_fft_mean"] = np.mean(fft_values)
                        window_features[f"{col}_fft_std"] = np.std(fft_values)

                        # 支配周波数
                        dominant_freq_idx = np.argmax(fft_values)
                        window_features[f"{col}_dominant_freq"] = (
                            dominant_freq_idx * self.sampling_rate / len(values)
                        )

                        # スペクトルエントロピー（簡易版）
                        psd = fft_values**2
                        psd_normalized = psd / np.sum(psd)
                        psd_normalized = psd_normalized[psd_normalized > 0]  # ゼロ除去
                        if len(psd_normalized) > 0:
                            window_features[f"{col}_spectral_entropy"] = -np.sum(
                                psd_normalized * np.log(psd_normalized)
                            )
                        else:
                            window_features[f"{col}_spectral_entropy"] = 0

            features_list.append(window_features)

        return pd.DataFrame(features_list)

    def extract_dynamic_features(
        self,
        df: pd.DataFrame,
        window_size_seconds: float = 5.0,
        overlap_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """動的特徴量の抽出"""
        if self.sampling_rate is None:
            self.sampling_rate = 1.0

        window_size = max(1, int(window_size_seconds * self.sampling_rate))
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        sensor_cols = [
            col
            for col in df.columns
            if col not in ["id", "target", "behavior", "timestamp"]
        ]

        features_list = []

        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i : i + window_size]
            window_features = {}

            if "id" in df.columns:
                window_features["id"] = window_data["id"].iloc[0]
            if "target" in df.columns:
                window_features["target"] = (
                    window_data["target"].mode().iloc[0]
                    if len(window_data["target"].mode()) > 0
                    else 0
                )

            for col in sensor_cols:
                if col in window_data.columns:
                    values = window_data[col].values

                    # 1次差分（速度相当）
                    diff1 = np.diff(values)
                    if len(diff1) > 0:
                        window_features[f"{col}_diff1_mean"] = np.mean(diff1)
                        window_features[f"{col}_diff1_std"] = np.std(diff1)
                        window_features[f"{col}_diff1_max"] = np.max(np.abs(diff1))
                        window_features[f"{col}_diff1_rms"] = np.sqrt(np.mean(diff1**2))

                    # 2次差分（加速度相当）
                    if len(diff1) > 1:
                        diff2 = np.diff(diff1)
                        window_features[f"{col}_diff2_mean"] = np.mean(diff2)
                        window_features[f"{col}_diff2_std"] = np.std(diff2)
                        window_features[f"{col}_diff2_max"] = np.max(np.abs(diff2))
                        window_features[f"{col}_diff2_rms"] = np.sqrt(np.mean(diff2**2))

                    # ゼロ交差回数
                    zero_crossings = np.sum(
                        np.diff(np.sign(values - np.mean(values))) != 0
                    )
                    window_features[f"{col}_zero_crossings"] = zero_crossings

            features_list.append(window_features)

        return pd.DataFrame(features_list)


class ExperimentRunner:
    """実験実行管理クラス"""

    def __init__(self, data_path: str, results_dir: str = "results"):
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.extractor = OptimizedFeatureExtractor()
        self.experiment_results = {}

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """データの読み込み"""
        print("データを読み込み中...")

        # パーケットファイルとして読み込み
        if self.data_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        print(f"データサイズ: {df.shape}")

        # behaviorをバイナリ値に変換
        if "behavior" in df.columns:
            print(f"behavior分布: {df['behavior'].value_counts().to_dict()}")
            # BFRB行動を1、その他を0とする（仮の分類）
            bfrb_behaviors = ["Performs gesture"]  # 実際のBFRB行動
            df["behavior_binary"] = df["behavior"].apply(
                lambda x: 1 if x in bfrb_behaviors else 0
            )
            binary_dist = df['behavior_binary'].value_counts().to_dict()
            print(f"バイナリbehavior分布: {binary_dist}")

        # trainとtestに分割（仮）
        if "behavior_binary" in df.columns:
            # 80/20で分割
            train_size = int(0.8 * len(df))
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()
        else:
            train_df = df.copy()
            test_df = df.copy()

        return train_df, test_df

    def experiment_1a_window_optimization(
        self, train_df: pd.DataFrame
    ) -> dict[str, Any]:
        """実験1A: 最適時間窓パラメータ決定"""
        print("\n=== 実験1A: 最適時間窓パラメータ決定 ===")
        print(f"データサイズ: {train_df.shape}")
        print(f"behavior分布: {train_df['behavior'].value_counts().to_dict()}")
        binary_dist = train_df['behavior_binary'].value_counts().to_dict()
        print(f"behavior_binary分布: {binary_dist}")

        # パラメータ候補（サンプル数ベース、より小さな値で開始）
        window_sizes = [20, 50, 100]  # サンプル数
        overlap_ratios = [0.5, 0.75]

        best_params = None
        best_score = 0
        results = {}

        # データのサブサンプル（高速化、但し複数シーケンスを含むように）
        sample_sequences = train_df["sequence_id"].unique()[:20]  # 最初の20シーケンス
        sample_df = train_df[
            train_df["sequence_id"].isin(sample_sequences)
        ].reset_index(drop=True)
        print(f"サンプルデータサイズ: {sample_df.shape}")
        sample_binary_dist = sample_df['behavior_binary'].value_counts().to_dict()
        print(f"サンプルbehavior_binary分布: {sample_binary_dist}")

        for window_size in window_sizes:
            for overlap_ratio in overlap_ratios:
                param_key = f"window_{window_size}samples_overlap_{overlap_ratio}"
                print(f"テスト中: {param_key}")

                try:
                    # 特徴量抽出
                    features_df = self.extractor.extract_time_window_features(
                        sample_df, window_size, overlap_ratio
                    )

                    if len(features_df) < 10:  # 最小サンプル数チェック
                        print(f"  スキップ: サンプル数不足 ({len(features_df)})")
                        continue

                    # 特徴量とターゲットを準備
                    X = features_df[self.extractor.feature_names_].fillna(0)
                    y = (
                        features_df["target"]
                        if "target" in features_df.columns
                        else np.zeros(len(X))
                    )

                    # 無限値やNaN値のチェック
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

                    print(f"  特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}")
                    print(f"  target分布: {np.bincount(y.astype(int))}")

                    if len(np.unique(y)) < 2:  # バイナリ分類チェック
                        print("  スキップ: ターゲットの多様性不足")
                        continue

                    # 簡易モデルで評価
                    model = LightGBMModel(n_estimators=50, verbose=-1, random_state=42)

                    # クロスバリデーション
                    cv_scores = []
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                    for train_idx, val_idx in skf.split(X, y):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                        model.train(X_train.values, y_train.values)
                        predictions = model.predict(X_val.values)
                        score = accuracy_score(y_val, predictions)
                        cv_scores.append(score)

                    avg_score = np.mean(cv_scores)

                    results[param_key] = {
                        "window_size": window_size,
                        "overlap_ratio": overlap_ratio,
                        "cv_score": avg_score,
                        "cv_std": np.std(cv_scores),
                        "n_features": len(self.extractor.feature_names_),
                        "n_samples": len(features_df),
                    }

                    print(f"  精度: {avg_score:.4f} ± {np.std(cv_scores):.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = (window_size, overlap_ratio)

                except Exception as e:
                    print(f"  エラー: {str(e)}")
                    continue

        if best_params:
            print(
                f"\n最適パラメータ: window_size={best_params[0]}samples, "
                f"overlap_ratio={best_params[1]}"
            )
            print(f"最良スコア: {best_score:.4f}")
        else:
            print("\n警告: 有効な結果が得られませんでした。デフォルト値を使用します。")
            best_params = (100, 0.5)

        return {
            "best_window_size": best_params[0] if best_params else 100,
            "best_overlap_ratio": best_params[1] if best_params else 0.5,
            "best_score": best_score,
            "all_results": results,
        }

    def run_experiments(self):
        """全実験を実行"""
        print("特徴量エンジニアリング実験を開始...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # データ読み込み
        train_df, test_df = self.load_data()

        # 実験1A: 最適時間窓パラメータ決定
        exp1a_results = self.experiment_1a_window_optimization(train_df)
        self.experiment_results["experiment_1a"] = exp1a_results

        # 結果保存
        results_file = (
            self.results_dir / f"feature_engineering_optimized_{timestamp}.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False)

        print(f"\n実験結果を保存: {results_file}")

        # サマリー表示
        print("\n=== 実験サマリー ===")
        print("実験1A - 最適窓パラメータ:")
        print(f"  窓サイズ: {exp1a_results['best_window_size']}秒")
        print(f"  重複率: {exp1a_results['best_overlap_ratio']}")
        print(f"  精度: {exp1a_results['best_score']:.4f}")


def main():
    """メイン実行関数"""
    # データパス
    data_path = Path("data/train.parquet")

    if not data_path.exists():
        print(f"データファイルが見つかりません: {data_path}")
        return

    # 実験実行
    runner = ExperimentRunner(str(data_path))
    runner.run_experiments()


if __name__ == "__main__":
    main()
