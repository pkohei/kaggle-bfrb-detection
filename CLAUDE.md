# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

これは **Kaggle競技プロジェクト** で、GitHub IssueとPull Requestを活用した構造化実験ワークフローによるBFRB（Body-Focused Repetitive Behaviors）検出を行います：
- **uv**: 高速な依存関係解決と管理のためのモダンなPythonパッケージマネージャー
- **Docker**: コンテナ化された開発環境
- **Dev Containers**: シームレスなコンテナベース開発のためのVS Code統合
- **CUDA対応**: 機械学習ワークロードのためのGPUコンピューティングサポート

### プロジェクトの説明

**Kaggle競技**: センサーデータを使用したBody-focused repetitive behaviors検出
- **目標**: 行動パターンの多クラス分類
- **データ**: 4つの行動クラスを持つ時系列センサーデータ
- **評価**: テストセットでの分類精度
- **ワークフロー**: PRベースの結果レビューを伴うIssue駆動実験開発

## 開発環境

このプロジェクトは、マシン間での一貫した開発環境のためにDevContainerを使用します。

### クイックスタート

1. VS Codeでプロジェクトを開く
2. コマンドパレットから「Dev Containers: Reopen in Container」を実行
3. 依存関係をインストール: `uv sync`
4. Kaggle認証情報を設定してデータをダウンロード: `uv run python scripts/setup_kaggle.py`
5. ベースラインモデルを作成: `uv run python scripts/create_quick_baseline.py`

## Kaggle競技コマンド

### セットアップとデータ管理

```bash
# Kaggle環境のセットアップと競技データのダウンロード
uv run python scripts/setup_kaggle.py

# プロジェクトの状況とデータサマリーの確認
uv run python scripts/project_summary.py
```

### モデル開発

```bash
# クイックベースラインモデルの作成（高速イテレーション推奨）
uv run python scripts/create_quick_baseline.py

# 複数アルゴリズムによる包括的ベースラインの作成
uv run python scripts/create_baseline.py
```

### パッケージ管理

```bash
# 依存関係のインストール（devグループも自動的に含まれる）
uv sync

# 新しいパッケージの追加
uv add package-name

# 開発パッケージの追加（dependency-groups使用）
uv add --group dev package-name

# 依存関係の更新
uv lock --upgrade
```

依存関係をインストールする際は、pyproject.tomlを直接編集するのではなく、`uv add`を使用してください。pyproject.tomlは自動的に更新されます。

### テスト

```bash
# 全テストの実行
uv run pytest

# カバレッジレポート付きでの実行
uv run pytest --cov

# 特定のテストファイルの実行
uv run pytest tests/test_main.py
```

### コード品質

```bash
# コードのフォーマット
uv run ruff format

# リンティングの実行
uv run ruff check

# リンティングエラーの自動修正
uv run ruff check --fix

# インポート順序の自動整理
uv run ruff check --select I --fix
```

### 型チェック

```bash
# 型チェックの実行
uv run mypy src
```

### pre-commitフック

```bash
# pre-commitフックのインストール
uv run pre-commit install

# 全ファイルでのフック実行
uv run pre-commit run --all-files
```

## プロジェクトアーキテクチャ

### ディレクトリ構造

- **ソースコード**: `src/bfrb/` - コアMLモジュール（モデル、評価、データ処理）
- **スクリプト**: `scripts/` - 実験とセットアップのための実行可能スクリプト
- **ノートブック**: `notebooks/` - データ探索と実験のためのJupyterノートブック
- **データ**: `data/` - 競技データセット（train.csv、test.csvなど）
- **提出**: `submissions/` - Kaggle用に生成された提出ファイル
- **結果**: `results/` - モデル評価結果とプロット
- **テスト**: `tests/` - pytestを使用したテストファイル
- **設定**: `pyproject.toml` - プロジェクト設定と依存関係
- **依存関係**: `uv.lock` - 再現可能なビルドのための依存関係ロックファイル
- **GitHubテンプレート**: `.github/` - 構造化ワークフローのためのIssueとPRテンプレート

### 主要技術

- **Python 3.12**: プログラミング言語
- **uv**: 高速Pythonパッケージマネージャー
- **scikit-learn, LightGBM, XGBoost**: 機械学習アルゴリズム
- **pandas, numpy**: データ操作と数値計算
- **matplotlib, seaborn, plotly**: データ可視化
- **pytest**: テストフレームワーク
- **ruff**: コードフォーマッターとリンター
- **mypy**: 静的型チェッカー
- **pre-commit**: コード品質のためのGitフック
- **Jupyter**: インタラクティブデータ分析
- **Kaggle API**: 競技データと提出管理

## 実験ワークフロー

### Issue駆動実験管理

**ユーザーから特に指定がない限り、実験開発では以下のワークフローに従ってください**

1. **実験計画（GitHub Issue）**
   - 実験テンプレートを使用してGitHub Issueを作成: `.github/ISSUE_TEMPLATE/experiment.md`
   - 仮説、手法、成功基準、予想スケジュールを定義
   - ベースライン比較と評価指標を含める
   - 実験アプローチについて承認/議論を得る

2. **実験セットアップ**
   - mainブランチをチェックアウトして最新の変更を取得: `git pull origin main`
   - 実験ブランチを作成: `git checkout -b experiment/[issue番号]-[簡潔な説明]`
   - 追跡可能性のためブランチ名にissue番号を含める

3. **実験開発**
   - Issueの計画に従って実験を実装
   - 再現性のため`scripts/`でスクリプトを作成または修正
   - 明確な結果とともにJupyterノートブックで実験を文書化
   - issueを参照する意味のあるメッセージでコミット

4. **実験実行と文書化**
   - 実験を実行して結果を収集
   - 可視化と性能指標を生成
   - 発見、洞察、次のステップを文書化
   - 該当する場合は提出ファイルを作成

5. **結果を含むPull Request**
   - 実験結果テンプレートを使用してPRを作成: `.github/PULL_REQUEST_TEMPLATE/experiment_results.md`
   - 実験結果、可視化、分析を含める
   - 主要な発見と推奨事項をまとめる
   - 文脈のため元のIssueにリンク

6. **レビューと統合**
   - レビューフィードバックと質問に対応
   - レビュワーと結果と影響について議論
   - 承認後にPRをマージし、関連Issueをクローズ
   - mainブランチを更新し、実験ブランチをクリーンアップ

### テンプレート

- **実験Issueテンプレート**: `.github/ISSUE_TEMPLATE/experiment.md`
- **実験PRテンプレート**: `.github/PULL_REQUEST_TEMPLATE/experiment_results.md`

### 言語規約

**IssueとPRは日本語で記述してください**
- タイトル、説明、コメントすべて日本語で記述
- コード内のコメントとdocstringも日本語推奨
- 変数名や関数名は英語のまま（Python慣例に従う）

## コード品質ガイドライン

### 実験コード基準

**本格コード** (src/bfrb/)：
- 完全な型ヒントと文書化
- 包括的なユニットテスト
- 100% ruff/mypy準拠
- 徹底したエラーハンドリング

**実験コード** (scripts/, notebooks/)：
- 主要関数の基本的な型ヒント
- メインワークフローの機能テスト
- ruffフォーマット必須、mypyの警告は許容
- 明確なコメントと文書化

**共通基準**:
- コードフォーマットには`ruff format`を使用
- ruffリンティングルールに従う（`ruff check`）
- コミット前に`mypy`を実行（実験では警告OK）
- Python命名規則に従う（PEP 8）
- パブリック関数とクラスにdocstringを記述

### 実験ベストプラクティス

- **再現性**: ランダムシードを設定し、すべてのパラメータを文書化
- **バージョン管理**: 明確で説明的なメッセージで実験コードをコミット
- **文書化**: 探索的分析と結果にJupyterノートブックを使用
- **性能追跡**: 検証スコアを監視し、実験履歴を追跡
- **リソース管理**: 計算要件と実行時間を文書化

### テスト要件

**本格コード**:
- 包括的なユニットテストを記述
- テストカバレッジ80%以上を維持
- エッジケースとエラー条件をテスト

**実験コード**:
- メイン実行パスをテスト
- データ処理パイプラインを検証
- 主要な結果の再現性を確保

## CUDA/GPU開発

このプロジェクトは、CUDA 12.4によるGPUコンピューティング用に設定されています。

### GPU情報

```bash
# GPUステータスの確認
nvidia-smi
```

### GPU対応トレーニング

```bash
# ほとんどのMLフレームワークは利用可能な場合自動的にGPUを使用
# トレーニング中のGPU使用量監視
watch -n 1 nvidia-smi
```

## トラブルシューティング

### 一般的な問題

1. **依存関係の競合**: `uv lock --upgrade`で解決
2. **インポートエラー**: 正しい仮想環境にいることを確認
3. **CUDAバージョンの不一致**: ホストCUDAバージョンの互換性を確認
4. **Kaggle APIエラー**: 認証情報と競技参加を確認
5. **メモリ問題**: 大きなデータセットにはデータサンプリングまたはバッチ処理を使用

### 実験の問題

1. **再現性の問題**: ランダムシードと環境の一貫性を確認
2. **性能の後退**: 同じ検証セットを使用してベースラインと比較
3. **リソース制約**: 実験中のメモリ/GPU使用量を監視
4. **データリーク**: 適切なtrain/validation/testの分割を確認

### ヘルプを得る

- uvドキュメントのレビュー: https://docs.astral.sh/uv/
- Kaggle競技ディスカッションフォーラムの確認
- scripts/README.mdで詳細なコマンド使用法をレビュー
- プロジェクト状況確認: `uv run python scripts/project_summary.py`
- 実験計画と議論のためのGitHub Issue作成
