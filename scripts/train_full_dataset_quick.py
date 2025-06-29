#!/usr/bin/env python3
"""
全データセット学習のクイック版（短時間実行）
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def main():
    """メイン処理"""
    print("🚀 全データセット学習実験（クイック版）")
    print("=" * 50)
    
    # データ読み込み
    print("Parquetファイルから読み込み中...")
    start_time = time.time()
    train_df = pd.read_parquet("data/train.parquet")
    test_df = pd.read_parquet("data/test.parquet")
    print(f"読み込み完了: {time.time() - start_time:.2f}秒")
    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")
    
    # ターゲット列を特定
    target_col = 'behavior'
    
    # 数値特徴量を選択
    feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    # 特徴量とターゲットを分離
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    X_test = test_df[feature_cols].values
    
    # ターゲット変数の数値化
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 欠損値処理
    X = np.nan_to_num(X, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    print(f"\n特徴量数: {len(feature_cols)}")
    print(f"クラス分布: {np.bincount(y)}")
    
    # 訓練・検証分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    
    # RandomForest（軽量版）
    print("\n=== RandomForest訓練 ===")
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,  # 200→100に削減
        max_depth=15,      # 20→15に削減
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    rf_pred = rf_model.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_pred)
    print(f"RandomForest精度: {rf_accuracy:.4f} (訓練時間: {rf_train_time:.1f}秒)")
    
    # LightGBM（軽量版）
    print("\n=== LightGBM訓練 ===")
    start_time = time.time()
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        n_estimators=100,  # 500→100に削減
        learning_rate=0.1,  # 0.05→0.1に増加
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_train_time = time.time() - start_time
    
    lgb_pred = lgb_model.predict(X_val)
    lgb_accuracy = accuracy_score(y_val, lgb_pred)
    print(f"LightGBM精度: {lgb_accuracy:.4f} (訓練時間: {lgb_train_time:.1f}秒)")
    
    # 提出ファイル作成
    best_model = rf_model if rf_accuracy > lgb_accuracy else lgb_model
    best_name = "RandomForest" if rf_accuracy > lgb_accuracy else "LightGBM"
    best_accuracy = max(rf_accuracy, lgb_accuracy)
    
    predictions = best_model.predict(X_test)
    
    # 元のラベルに戻す
    predictions_labels = le.inverse_transform(predictions)
    
    # 提出ファイル作成
    submission_df = pd.DataFrame({
        'id': range(len(predictions)),
        'behavior': predictions_labels
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/full_dataset_quick_{best_name.lower()}_{timestamp}.csv"
    submission_df.to_csv(submission_file, index=False)
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎉 実験完了サマリー")
    print("=" * 50)
    print(f"RandomForest精度: {rf_accuracy:.4f}")
    print(f"LightGBM精度: {lgb_accuracy:.4f}")
    print(f"最高精度: {best_name} ({best_accuracy:.4f})")
    print(f"提出ファイル: {submission_file}")
    
    # ベースライン比較
    baseline_accuracy = 0.7361
    improvement = best_accuracy - baseline_accuracy
    print(f"\nベースライン比較:")
    print(f"  前回: {baseline_accuracy:.4f}")
    print(f"  今回: {best_accuracy:.4f}")
    print(f"  改善: {improvement:+.4f} ({improvement/baseline_accuracy*100:+.1f}%)")
    
    # 結果を保存
    results = {
        'experiment': 'full_dataset_quick',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'baseline_accuracy': baseline_accuracy,
        'randomforest_accuracy': rf_accuracy,
        'lightgbm_accuracy': lgb_accuracy,
        'best_model': best_name,
        'best_accuracy': best_accuracy,
        'improvement': improvement,
        'improvement_percent': improvement/baseline_accuracy*100,
        'submission_file': submission_file
    }
    
    with open('results/full_dataset_quick_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✨ 実験完了！結果保存: results/full_dataset_quick_results.json")


if __name__ == "__main__":
    main()