# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## 原則

**あらゆる場面で以下の原則 (law) に必ず従うこと**

<law>
Claude原則
1. Claudeはあらゆる作業の実行前に必ずCLAUDE.mdを読み返す
2. Claudeはすべてのチャットの冒頭にClaude原則を逐語的に必ず画面出力してから対応する
</law>

## プロジェクト概要

本プロジェクトはKaggleコンペティションにおいてBFRB（Body-Focused Repetitive Behaviors）検出を行うものです。

URL: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data

## 実験ワークフロー

Claudeが実験するときはこのワークフローに従う必要がある。
実験ワークフローは3つのフェーズと、各フェーズ内の複数ステップから構成される。
Gemini CLIに依頼する際は「Gemini CLI 依頼ガイド」を参照すること。

### 1. 実験計画フェーズ

1. 技術調査をGemini CLIに依頼する
2. 調査結果に基づいて実験計画を考え、ユーザーに提示する
3. 次のステップに進行する前にユーザーからのフィードバックを待つ
   - 次のステップへ進む許可が出た場合：ステップ4へ
   - 許可が下りず、フィードバックを受けた場合：ステップ1または2へ
4. 実験テンプレート (`.github/ISSUE_TEMPLATE/experiment.md`) を使用してGitHub Issueを作成
5. ユーザーに実験計画が完了したことを伝える

### 2. 実験準備フェーズ

1. mainブランチをチェックアウトして最新の変更を取得: `git pull origin main`
2. 現在のブランチ状況を確認する
3. 必要に応じて実験ブランチを作成: `git checkout -b experiment/[issue番号]-[簡潔な説明]`
4. ユーザーに実験準備が完了したことを伝える

### 3. 実験フェーズ

1. Issueの計画にしたがって実験を実装し、実装内容をコミットする
2. 実装した実験コードについてGemini CLIに実験コードレビューを要求する
3. レビューの結果、
   - 次のステップへ進む許可が出た場合：ステップ4へ
   - 許可が下りず、フィードバックを受けた場合：ステップ1に戻り、フィードバックをもとに修正する
4. 実験コードを実行する
5. 出力された実験結果についてGemini CLIに実験結果レビューを要求する
6. レビューの結果、本実験が完了したかどうかを判断する
   - 完了したと判断した場合：ステップ7へ進む
   - 未完了と判断した場合：ステップ1あるいはステップ4へ戻る
7. 実験結果テンプレート (`.github/PULL_REQUEST_TEMPLATE/experiment_results.md`) を使用してPRを作成する
8. ユーザーに実験が完了し、PRが作成されたことを伝える


## コマンド

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


## Gemini CLI 依頼ガイド

### 概要

Gemini CLIは以下の特徴を持つAI Agentです。

- 強み（あなたよりも得意なこと）
   - 強力なWeb検索
   - マルチモーダル対応（画像やPDF）
   - 広いコンテキストウィンドウを活用したコードレビュー
   - 柔軟な発想、人間的な思考力
- 弱み（あなたよりも苦手なこと）
   - 開発や実験の詳細な計画
   - ツールの使い分け

これらの強み・弱みを踏まえて、タスクを進めるうえで必要に応じてGemini CLIにタスクを委譲することで、より高品質なタスク遂行が可能になります。**うまくGemini CLI を使いながらタスクを進めてください**


### Gemini CLIの基本機能

- 自律的なコマンド実行によるタスク実行
- リポジトリ内の読み込み・編集
- Web検索

### コマンド実行方法

```bash
gemini -p "指示内容"
```

### 注意点

- Gemini CLIは会話を保存せず、コマンドごとに記憶がリセットされる。一問一答となるよう心掛ける。
- Gemini CLIが自動で実行できるコマンドは限られるので、不完全な回答が返ってくることがある。その際は `gemini --yolo -p "指示内容"`であらゆるコマンドを実行できるようになるので完全な回答が返ってきやすくなる。

### タスク依頼テンプレート

- 技術調査テンプレート: `docs/gemini/investigation-request.md`
- コードレビューテンプレート: `docs/gemini/code-review-request.md`
- 実験結果レビューテンプレート: `docs/gemini/result-review-request.md`


### ドキュメント

Gemini CLIの使い方について悩んだときは下記のドキュメントを参照してください。

https://github.com/google-gemini/gemini-cli/blob/main/docs/index.md

## プロジェクトアーキテクチャ

- ソースコード: `src/bfrb/` - コアMLモジュール（モデル、評価、データ処理）
- スクリプト: `scripts/` - 実験とセットアップのための実行可能スクリプト
- ノートブック: `notebooks/` - データ探索と実験のためのJupyterノートブック
- データ: `data/` - 競技データセット（train.csv、test.csvなど）
- 提出: `submissions/` - Kaggle用に生成された提出ファイル
- 結果: `results/` - モデル評価結果とプロット
- テスト: `tests/` - pytestを使用したテストファイル
- 設定: `pyproject.toml` - プロジェクト設定と依存関係
- 依存関係: `uv.lock` - 再現可能なビルドのための依存関係ロックファイル
- GitHubテンプレート: `.github/` - 構造化ワークフローのためのIssueとPRテンプレート
