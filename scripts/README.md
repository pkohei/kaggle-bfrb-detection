# Scripts Directory

このディレクトリには、BFRB Detection プロジェクトの実行可能スクリプトが含まれています。

## 📋 利用可能なスクリプト

### 🚀 セットアップ・基本操作

#### `setup_kaggle.py`
Kaggle競技環境の初期セットアップを行います。

```bash
uv run python scripts/setup_kaggle.py
```

**機能:**
- Kaggle認証情報の設定確認
- プロジェクトディレクトリの作成
- 競技データのダウンロード
- セットアップの検証

**必要な条件:**
- Kaggle競技への参加
- `~/.kaggle/kaggle.json` の設置

---

#### `project_summary.py`
プロジェクトの現在の状況とサマリーを表示します。

```bash
uv run python scripts/project_summary.py
```

**出力内容:**
- データセット情報
- 作成済み提出ファイル一覧
- 最新モデルの性能
- 推奨次ステップ
- 便利なコマンド一覧

---

### 🤖 モデル構築

#### `create_quick_baseline.py`
高速なベースラインモデルを作成します（推奨）。

```bash
uv run python scripts/create_quick_baseline.py
```

**特徴:**
- **実行時間:** ~1-2分
- **データサンプル:** 50,000行
- **アルゴリズム:** RandomForest
- **特徴量:** 50個の数値特徴量
- **期待精度:** ~73%

**出力:**
- 検証精度の報告
- 特徴量重要度分析
- Kaggle提出用CSVファイル

---

#### `create_baseline.py`
完全なベースラインモデル（複数アルゴリズム）を作成します。

```bash
uv run python scripts/create_baseline.py
```

**特徴:**
- **実行時間:** ~10-20分
- **データ:** 全データセット
- **アルゴリズム:** LightGBM, XGBoost, RandomForest, Ensemble
- **特徴量:** 全数値特徴量

⚠️ **注意:** 大容量データのため実行時間が長くなります。

---

## 🗂️ 実行順序

初回セットアップ時は以下の順序で実行してください：

```bash
# 1. 環境セットアップ
uv run python scripts/setup_kaggle.py

# 2. クイックベースライン作成
uv run python scripts/create_quick_baseline.py

# 3. プロジェクト状況確認
uv run python scripts/project_summary.py

# 4. (オプション) 完全ベースライン作成
uv run python scripts/create_baseline.py
```

---

## 📁 出力ファイル

スクリプト実行により以下のファイルが生成されます：

### データディレクトリ (`data/`)
- `train.csv` - 訓練データ
- `test.csv` - テストデータ
- `train_demographics.csv` - 訓練者属性
- `test_demographics.csv` - テスト者属性

### 提出ファイルディレクトリ (`submissions/`)
- `quick_baseline_YYYYMMDD_HHMMSS.csv` - クイックベースライン
- `baseline_*_YYYYMMDD_HHMMSS.csv` - 各種ベースラインモデル

### 結果ディレクトリ (`results/`)
- `model_comparison.csv` - モデル性能比較
- `confusion_matrix_*.png` - 混同行列図
- `feature_importance_*.csv` - 特徴量重要度

---

## ⚡ クイックスタート

最小限の手順でベースラインモデルを作成：

```bash
# Kaggle認証設定後
uv run python scripts/setup_kaggle.py
uv run python scripts/create_quick_baseline.py
```

この2つのコマンドで、すぐにKaggleに提出可能なベースラインモデルが完成します！

---

## 🔧 トラブルシューティング

### Kaggle認証エラー
```bash
OSError: Could not find kaggle.json
```

**解決方法:**
1. Kaggle競技に参加
2. kaggle.jsonを `~/.kaggle/` に配置
3. `chmod 600 ~/.kaggle/kaggle.json` で権限設定

### メモリ不足エラー
```bash
MemoryError: Unable to allocate array
```

**解決方法:**
- `create_quick_baseline.py` を使用（サンプリング済み）
- サンプルサイズを減らす（スクリプト内のsample_size変数を編集）

### 長時間実行
- `create_baseline.py` は大容量データのため時間がかかります
- 初回は `create_quick_baseline.py` の使用を推奨
