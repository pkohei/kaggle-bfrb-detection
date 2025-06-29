# Kaggle BFRB Detection

**Body-Focused Repetitive Behaviors (BFRB) 検出プロジェクト**

Kaggle競技における時系列センサーデータを使用したBFRB（身体集中反復行動）の多クラス分類タスクです。

## 🏆 競技情報

- **競技名**: [Child Mind Institute — Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- **目標**: センサーデータから4つの行動クラスを分類
- **評価**: テストセットでの分類精度
- **データ**: 時系列センサーデータ

## 🚀 クイックスタート

### 1. 開発環境の起動

```bash
# VS Codeでプロジェクトを開く
# コマンドパレット (Ctrl+Shift+P) から「Dev Containers: Reopen in Container」を実行
```

### 2. 依存関係のインストール

```bash
uv sync
```

### 3. Kaggle認証とデータダウンロード

```bash
uv run python scripts/setup_kaggle.py
```

### 4. ベースラインモデルの作成

```bash
# 高速ベースライン（推奨）
uv run python scripts/create_quick_baseline.py

# 包括的ベースライン
uv run python scripts/create_baseline.py
```

## 📁 プロジェクト構造

```
kaggle-bfrb-detection/
├── src/bfrb/              # コア機械学習モジュール
├── scripts/               # 実験・セットアップスクリプト
├── notebooks/             # Jupyter分析ノートブック
├── data/                  # 競技データセット
├── submissions/           # Kaggle提出ファイル
├── results/               # モデル評価結果
├── tests/                 # テストファイル
├── .github/               # GitHub テンプレート
├── pyproject.toml         # プロジェクト設定
└── uv.lock               # 依存関係ロック
```

## 🔬 実験ワークフロー

### Issue駆動実験管理

1. **実験計画** - GitHub Issueで仮説と手法を定義
2. **ブランチ作成** - `experiment/[issue番号]-[説明]`
3. **実験実装** - スクリプトとノートブックで開発
4. **結果Pull Request** - 可視化と分析を含むPR作成
5. **レビューと統合** - 結果議論後にマージ

### テンプレート

- 実験Issue: `.github/ISSUE_TEMPLATE/experiment.md`
- 実験PR: `.github/PULL_REQUEST_TEMPLATE/experiment_results.md`

## 🛠️ よく使うコマンド

### データとモデル

```bash
# プロジェクト状況確認
uv run python scripts/project_summary.py

# クイックベースライン作成
uv run python scripts/create_quick_baseline.py

# 包括的ベースライン作成
uv run python scripts/create_baseline.py
```

### 開発ツール

```bash
# テスト実行
uv run pytest

# コードフォーマット
uv run ruff format

# リンティング
uv run ruff check --fix

# 型チェック
uv run mypy src
```

### パッケージ管理

```bash
# パッケージ追加
uv add package-name

# 開発パッケージ追加
uv add --dev package-name

# 依存関係更新
uv lock --upgrade
```

## 🖥️ 開発環境

### 必要要件

- Docker
- Visual Studio Code
- Dev Containers extension
- NVIDIA Docker (GPU使用時)
- NVIDIA GPU drivers

### 技術スタック

- **Python 3.12** - メイン言語
- **uv** - 高速パッケージマネージャー
- **scikit-learn, LightGBM, XGBoost** - 機械学習
- **pandas, numpy** - データ処理
- **matplotlib, seaborn, plotly** - 可視化
- **pytest** - テスト
- **Jupyter** - データ分析

### GPU/CUDA サポート

```bash
# GPU情報確認
nvidia-smi

# GPU使用量監視
watch -n 1 nvidia-smi
```

## 📋 コード品質基準

### 本格コード (src/)
- 完全な型ヒントと文書化
- 包括的なテスト (80%以上カバレッジ)
- 100% ruff/mypy準拠

### 実験コード (scripts/, notebooks/)
- 基本的な型ヒント
- 機能テスト
- ruffフォーマット必須

## 🔧 トラブルシューティング

### よくある問題

- **依存関係競合**: `uv lock --upgrade`
- **インポートエラー**: 仮想環境確認
- **Kaggle APIエラー**: 認証情報確認
- **CUDA問題**: ホストCUDAバージョン確認

### ヘルプリソース

- [uv ドキュメント](https://docs.astral.sh/uv/)
- [Kaggle競技ディスカッション](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion)
- プロジェクト状況: `uv run python scripts/project_summary.py`

## 📄 ライセンス

MIT License
