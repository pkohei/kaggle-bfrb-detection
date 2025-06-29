# Gemini プロジェクトコンテキスト: Kaggle BFRB検出

このドキュメントは、Geminiが開発を支援するための「Kaggle BFRB検出」プロジェクトのコンテキストを提供します。

## 0. Geminiが本プロジェクトで果たすべき役割

あなたには以下の2つの役割があります。

1. 技術調査員
    - データサイエンス、機械学習、数学、その他科学技術に長けた調査員として徹底的に技術を調査してユーザーの実験をサポートする
2. 実験レビュワー
    - データサイエンス、機械学習、数学などの科学技術とソフトウェア工学に長けたリーダーとして、実験コードや実験結果をレビューして適切なアドバイスを行う

## 1. プロジェクト概要

これは、Body-Focused Repetitive Behaviors（BFRB）検出に焦点を当てたKaggleコンペティション用の機械学習プロジェクトです。目標は、時系列センサーデータに対してマルチクラス分類を実行し、4つの異なる行動クラスを識別することです。

- **コンペティション:** [Child Mind Institute — Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- **コアタスク:** 時系列分類
- **主要評価指標:** 分類精度

## 2. プロジェクト構造

- `src/bfrb/`: コア、本番品質の機械学習ソースコード
- `scripts/`: 実験、データ処理、ユーティリティのためのスクリプト
- `notebooks/`: 探索と分析のためのJupyterノートブック
- `data/`: 生データと処理済みコンペティションデータ
- `submissions/`: Kaggle提出用に生成されたファイル
- `results/`: モデル評価出力とサマリー
- `tests/`: `src`コードのユニット・統合テスト
- `pyproject.toml`: `uv`で管理されたプロジェクトメタデータと依存関係
- `compose.yml`: 開発環境用Docker Compose設定
