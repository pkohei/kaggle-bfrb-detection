#!/usr/bin/env python3
"""
高度統計特徴量抽出モジュール

Gemini調査結果に基づく手動実装アプローチ：
- pandas/numpy/scipyを活用した効率的な特徴量生成
- tsfreshの代替としてPython 3.12完全対応
- GroupKFold/sequence_id基準の処理に最適化
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq


class AdvancedFeatureExtractor:
    """高度統計・周波数特徴量抽出器（手動実装版）"""

    def __init__(
        self,
        sampling_rate: float = 20.0,
        n_fft_features: int = 5,
        meta_columns: list[str] | None = None,
    ):
        """
        Parameters:
        -----------
        sampling_rate : float
            サンプリングレート（Hz）
        n_fft_features : int
            FFTから抽出する主要特徴数
        meta_columns : List[str]
            メタデータ列名リスト
        """
        self.sampling_rate = sampling_rate
        self.n_fft_features = n_fft_features
        self.feature_names_: list[str] = []

        self.meta_columns = meta_columns or [
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

    def extract_statistical_features(
        self, df: pd.DataFrame, group_col: str = "sequence_id"
    ) -> pd.DataFrame:
        """
        統計的特徴量を抽出

        Parameters:
        -----------
        df : pd.DataFrame
            時系列データ（センサー列を含む）
        group_col : str
            グループ化する列名

        Returns:
        --------
        pd.DataFrame
            統計特徴量
        """
        # センサー列を特定
        sensor_cols = [col for col in df.columns if col not in self.meta_columns]

        # 統計関数のリスト（pandasのagg形式）
        agg_funcs = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "var",
            lambda x: stats.skew(x, nan_policy="omit"),
            lambda x: stats.kurtosis(x, nan_policy="omit"),
            lambda x: np.sqrt(np.mean(np.square(x))),
            lambda x: stats.iqr(x, nan_policy="omit"),
            lambda x: np.ptp(x),
            lambda x: stats.sem(x, nan_policy="omit"),
            lambda x: stats.variation(x, nan_policy="omit"),
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75),
        ]

        # 関数名のマッピング
        func_names = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "var",
            "skewness",
            "kurtosis",
            "rms",
            "iqr",
            "range",
            "sem",
            "cv",
            "q25",
            "q75",
        ]

        # グループ化して統計特徴量を計算
        statistical_features = df.groupby(group_col)[sensor_cols].agg(agg_funcs)  # type: ignore[arg-type]

        # カラム名を平坦化 (sensor_0_mean, sensor_0_std, ...)
        new_columns = []
        for sensor_col in sensor_cols:
            for func_name in func_names:
                new_columns.append(f"{sensor_col}_{func_name}")
        statistical_features.columns = pd.Index(new_columns)

        return statistical_features.reset_index()

    def extract_frequency_features(
        self, df: pd.DataFrame, group_col: str = "sequence_id"
    ) -> pd.DataFrame:
        """
        周波数領域特徴量を抽出

        Parameters:
        -----------
        df : pd.DataFrame
            時系列データ
        group_col : str
            グループ化する列名

        Returns:
        --------
        pd.DataFrame
            周波数特徴量
        """
        # センサー列を特定
        sensor_cols = [col for col in df.columns if col not in self.meta_columns]

        def compute_fft_features(series: pd.Series) -> np.ndarray:
            """単一時系列のFFT特徴量計算"""
            values = series.dropna().values
            N = len(values)

            if N < 4:  # FFTに必要な最小データ点数
                return np.zeros(self.n_fft_features * 3)  # 振幅、周波数、パワー

            # FFT計算
            yf = fft(values)
            xf = fftfreq(N, 1 / self.sampling_rate)

            # 正の周波数のみ使用
            positive_freq_mask = xf > 0
            abs_yf = np.abs(yf[positive_freq_mask])
            freq_values = xf[positive_freq_mask]

            if len(abs_yf) == 0:
                return np.zeros(self.n_fft_features * 3)

            # パワースペクトル密度
            power = abs_yf**2

            # 上位N個の特徴を抽出
            top_indices = np.argsort(abs_yf)[::-1][: self.n_fft_features]

            # パディングで長さを統一
            top_amplitudes = np.zeros(self.n_fft_features)
            top_frequencies = np.zeros(self.n_fft_features)
            top_powers = np.zeros(self.n_fft_features)

            valid_length = min(len(top_indices), self.n_fft_features)
            top_amplitudes[:valid_length] = abs_yf[top_indices[:valid_length]]
            top_frequencies[:valid_length] = freq_values[top_indices[:valid_length]]
            top_powers[:valid_length] = power[top_indices[:valid_length]]

            return np.concatenate([top_amplitudes, top_frequencies, top_powers])

        # 各センサー、各グループでFFT特徴量を計算
        fft_features_list = []

        for sensor in sensor_cols:
            sensor_fft = (
                df.groupby(group_col)[sensor]
                .apply(compute_fft_features)
                .apply(pd.Series)
            )

            # カラム名を設定
            feature_names = []
            for i in range(self.n_fft_features):
                feature_names.extend(
                    [
                        f"{sensor}_fft_amp_{i}",
                        f"{sensor}_fft_freq_{i}",
                        f"{sensor}_fft_power_{i}",
                    ]
                )

            sensor_fft.columns = feature_names  # type: ignore[assignment]
            fft_features_list.append(sensor_fft)

        # 全センサーのFFT特徴量を結合
        fft_features = pd.concat(fft_features_list, axis=1)
        return fft_features.reset_index()

    def extract_correlation_features(
        self, df: pd.DataFrame, group_col: str = "sequence_id", max_pairs: int = 20
    ) -> pd.DataFrame:
        """
        センサー間相関特徴量を抽出（グローバル固定ペア方式）

        Gemini指摘対応: 各シーケンスで異なるペアが選ばれるバイアスを修正。
        全データで最も相関の高いペアを固定し、一貫性のある特徴量を生成。

        Parameters:
        -----------
        df : pd.DataFrame
            時系列データ
        group_col : str
            グループ化する列名
        max_pairs : int
            最大相関ペア数（計算量制御）

        Returns:
        --------
        pd.DataFrame
            相関特徴量
        """
        # センサー列を特定
        sensor_cols = [col for col in df.columns if col not in self.meta_columns]

        # Step 1: 全データでグローバル相関行列を計算
        print(f"グローバル相関行列を計算中（{len(sensor_cols)}センサー）...")
        all_sensor_data = df[sensor_cols]

        if len(all_sensor_data) < 3:
            # データ不足の場合はゼロ特徴量を返す
            empty_features = pd.DataFrame(
                {
                    group_col: df[group_col].unique(),
                    **{f"corr_{i}": 0.0 for i in range(max_pairs)},
                }
            )
            return empty_features

        global_corr_matrix = all_sensor_data.corr()

        # Step 2: 最も相関の高い（絶対値）上位ペアを固定選択
        sensor_pairs = []
        correlations = []

        for i in range(len(sensor_cols)):
            for j in range(i + 1, len(sensor_cols)):
                corr_value = global_corr_matrix.iloc[i, j]
                if not pd.isna(corr_value):
                    sensor_pairs.append((sensor_cols[i], sensor_cols[j]))
                    correlations.append(corr_value)

        # 絶対値で上位ペアを選択
        abs_correlations = np.abs(np.array(correlations))  # type: ignore[arg-type]
        sorted_indices = np.argsort(abs_correlations)[::-1]  # type: ignore[arg-type]
        top_pairs = [sensor_pairs[idx] for idx in sorted_indices[:max_pairs]]

        print(f"選択された上位{len(top_pairs)}ペア:")
        for i, (sensor1, sensor2) in enumerate(top_pairs[:5]):  # 上位5ペアを表示
            corr_val = correlations[sorted_indices[i]]
            print(f"  corr_{i}: {sensor1} - {sensor2} (r={corr_val:.3f})")

        # Step 3: 各シーケンスで固定ペアの相関を計算
        def compute_fixed_correlations(group_df: pd.DataFrame) -> pd.Series:
            """固定ペアの相関を計算"""
            sensor_data = group_df[sensor_cols]

            if len(sensor_data) < 3:
                return pd.Series(
                    [0.0] * max_pairs, index=[f"corr_{i}" for i in range(max_pairs)]
                )

            group_corr_matrix = sensor_data.corr()
            result_correlations = []

            for sensor1, sensor2 in top_pairs:
                try:
                    if (
                        sensor1 in group_corr_matrix.columns
                        and sensor2 in group_corr_matrix.columns
                    ):
                        corr_val = group_corr_matrix.loc[sensor1, sensor2]
                        if np.isnan(corr_val):
                            corr_val = 0.0
                    else:
                        corr_val = 0.0
                    result_correlations.append(corr_val)
                except (KeyError, IndexError):
                    result_correlations.append(0.0)

            # パディングで長さを統一
            while len(result_correlations) < max_pairs:
                result_correlations.append(0.0)

            return pd.Series(  # type: ignore[no-any-return]
                result_correlations[:max_pairs],
                index=[f"corr_{i}" for i in range(max_pairs)],
            )

        # グループごとに固定ペアの相関特徴量を計算
        correlation_features = df.groupby(group_col).apply(compute_fixed_correlations)  # type: ignore[no-any-return]
        return correlation_features.reset_index()

    def extract_all_features(
        self, df: pd.DataFrame, group_col: str = "sequence_id"
    ) -> pd.DataFrame:
        """
        全特徴量を統合して抽出

        Parameters:
        -----------
        df : pd.DataFrame
            時系列データ
        group_col : str
            グループ化する列名

        Returns:
        --------
        pd.DataFrame
            統合特徴量
        """
        print("統計特徴量を抽出中...")
        statistical = self.extract_statistical_features(df, group_col)

        print("周波数特徴量を抽出中...")
        frequency = self.extract_frequency_features(df, group_col)

        print("相関特徴量を抽出中...")
        correlation = self.extract_correlation_features(df, group_col)

        # 全特徴量を結合
        print("特徴量を統合中...")
        features = statistical.merge(frequency, on=group_col, how="inner")
        features = features.merge(correlation, on=group_col, how="inner")

        # 特徴量名を記録
        self.feature_names_ = [col for col in features.columns if col != group_col]

        print(f"特徴量抽出完了: {len(self.feature_names_)}次元")
        return features

    def get_feature_importance_summary(self) -> dict[str, int]:
        """特徴量タイプ別の次元数サマリーを取得"""
        summary = {
            "statistical": 0,
            "frequency": 0,
            "correlation": 0,
        }

        for feature_name in self.feature_names_:
            if any(
                stat in feature_name
                for stat in ["mean", "std", "skew", "kurt", "rms", "iqr"]
            ):
                summary["statistical"] += 1
            elif "fft" in feature_name:
                summary["frequency"] += 1
            elif "corr" in feature_name:
                summary["correlation"] += 1

        return summary
