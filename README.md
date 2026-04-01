# 社内RAGアシスタント

Ollama + ChromaDB を使った完全ローカルの RAG（検索拡張生成）ウェブアプリです。  
社内ドキュメントをアップロードし、自然言語で質問すると、ドキュメントに基づいた回答が得られます。

**社外にデータを送信しません。** すべてローカルで完結します。

## 特徴

- ドキュメントをアップロードするだけで即座に検索対象に
- PDF / Word(.docx) / テキスト(.txt) に対応
- 回答にはソース（参照元ファイル）を表示
- ストリーミング応答でリアルタイムに回答を表示
- チャット履歴を保持し、文脈を踏まえた追加質問が可能
- フォルダ一括登録スクリプト付き

## アーキテクチャ

```
[ドキュメント登録]
ファイル → テキスト抽出 → チャンク分割(500文字) → Embedding(nomic-embed-text) → ChromaDB

[質問回答]
質問 → Embedding → ChromaDB で類似チャンク検索(Top3)
                         ↓
         チャンクを参考情報としてプロンプトに注入
                         ↓
              qwen2.5:7b が回答を生成（ストリーミング）
```

| コンポーネント | 技術 |
|---|---|
| LLM | Ollama (qwen2.5:7b) |
| Embedding | Ollama (nomic-embed-text) |
| ベクトルDB | ChromaDB |
| バックエンド | FastAPI (Python) |
| フロントエンド | HTML / CSS / JavaScript |

## 必要環境

- **Python** 3.11 以上
- **Ollama** ([https://ollama.com](https://ollama.com))
- **GPU**: NVIDIA RTX 3070 Ti (8GB VRAM) 以上を推奨

## セットアップ

### 1. Ollama のモデル取得

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

> モデルの保存先を変更したい場合は、環境変数 `OLLAMA_MODELS` にパスを設定してください。  
> 例: `OLLAMA_MODELS=D:\llama`

### 2. Python 仮想環境の構築

```powershell
cd D:\github-copilot\ollama
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. サーバー起動

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

ブラウザで **http://localhost:8000** にアクセスしてください。

## 使い方

### Web UI からドキュメント登録

1. http://localhost:8000 にアクセス
2. ドラッグ＆ドロップまたはクリックでファイルを選択
3. 対応形式: `.txt` / `.pdf` / `.docx`

### Web UI から質問

チャット欄に質問を入力して送信するだけです。  
回答はストリーミングでリアルタイム表示されます。  
前の会話の文脈を踏まえた追加質問も可能です。

### フォルダ一括登録（bulk_import.py）

大量のドキュメントを一括で登録する場合に使います。サブフォルダも再帰的に探索します。

```powershell
# 基本: フォルダを指定して一括登録
python bulk_import.py ./docs

# 絶対パスでもOK
python bulk_import.py "D:\社内文書\技術ドキュメント"

# DB を初期化してから登録（やり直したい場合）
python bulk_import.py ./docs --reset

# 追加分だけ登録（--reset なしで差分追加）
python bulk_import.py "D:\社内文書\2026年3月"
```

出力例:

```
対象フォルダ: D:\社内文書\技術ドキュメント
対象ファイル: 150 件
対応形式: .docx, .pdf, .txt

  [1/150] ✅ 設計書/api-spec.pdf — 12 チャンク
  [2/150] ✅ 運用/障害対応手順.docx — 4 チャンク
  ...

==================================================
完了: 45.2 秒
成功: 148/150 ファイル
チャンク数: 1520
DB合計チャンク数: 1520
```

### DB リセット

Web UI のヘッダー右上の「DB リセット」ボタン、または一括登録時の `--reset` オプションで全ドキュメントを削除できます。

## API エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/upload` | ファイルアップロード（multipart/form-data） |
| `POST` | `/api/ask` | 質問（JSON: `{question, history}` → 一括回答） |
| `POST` | `/api/ask/stream` | 質問（JSON: `{question, history}` → SSE ストリーミング） |
| `GET` | `/api/stats` | DB 統計（登録チャンク数） |
| `POST` | `/api/reset` | DB 全削除 |

## 設定（環境変数）

| 環境変数 | デフォルト値 | 説明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama サーバーの URL |
| `CHAT_MODEL` | `qwen2.5:7b` | チャットに使用するモデル |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding に使用するモデル |
| `CHROMA_DIR` | `./chroma_data` | ChromaDB のデータ保存先 |
| `CHUNK_SIZE` | `500` | チャンク分割の文字数 |
| `CHUNK_OVERLAP` | `50` | チャンク間のオーバーラップ文字数 |
| `TOP_K` | `3` | 検索で取得するチャンク数 |

## プロジェクト構成

```
ollama/
├── app/
│   ├── __init__.py
│   ├── config.py            # 設定（環境変数から読み込み）
│   ├── main.py              # FastAPI エンドポイント
│   ├── ollama_client.py     # Ollama API クライアント（Embedding / Chat / Streaming）
│   ├── parser.py            # ファイル解析（txt / pdf / docx）
│   └── vectorstore.py       # ChromaDB 操作（登録 / 検索 / リセット）
├── static/
│   └── index.html           # フロントエンド UI
├── docs/                    # サンプルドキュメント
├── bulk_import.py           # フォルダ一括登録スクリプト
├── Caddyfile                # Caddy リバースプロキシ設定
├── requirements.txt
└── .gitignore
```

## GPU メモリ使用量（参考）

RTX 3070 Ti (8GB VRAM) での実測:

| モデル | サイズ | VRAM 使用量 |
|---|---|---|
| qwen2.5:7b | 4.7 GB | ~5-6 GB |
| nomic-embed-text | 274 MB | ~0.3 GB |
| **合計** | | **~6 GB** |

8GB VRAM に余裕で収まります。

## LLM の差し替え（より高性能なモデルへ）

環境変数 `CHAT_MODEL` を変更するだけで、コード修正なしにモデルを差し替えられます。

### 手順

```powershell
# 1. 新しいモデルを取得
ollama pull <モデル名>

# 2. 環境変数を設定してサーバー起動
$env:CHAT_MODEL = "<モデル名>"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### GPU 別おすすめモデル

| GPU (VRAM) | モデル | サイズ | 特徴 |
|---|---|---|---|
| **8GB** (RTX 3070 Ti 等) | `qwen2.5:7b` | 4.7 GB | 当プロジェクトのデフォルト。日本語十分実用的 |
| **8GB** | `gemma3:4b` | 3.3 GB | 軽量でレスポンスが速い |
| **12GB** (RTX 3060 12GB 等) | `qwen2.5:14b` | 9.0 GB | 7B より回答品質が向上 |
| **16GB** (RTX 4080 等) | `qwen2.5:32b-q4_K_M` | ~18 GB | 大幅に高品質。長文の理解力が高い |
| **24GB** (RTX 4090 等) | `qwen2.5:32b` | 20 GB | 32B フル精度 |
| **24GB** | `deepseek-r1:32b` | 20 GB | 推論特化。複雑な質問に強い |

### 差し替え例: qwen2.5:14b に変更

```powershell
ollama pull qwen2.5:14b
$env:CHAT_MODEL = "qwen2.5:14b"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Embedding モデルの差し替え

Embedding モデルも同様に差し替え可能です。ただし **変更後は DB リセット＋再登録が必要** です（ベクトルの次元数が異なるため）。

```powershell
ollama pull mxbai-embed-large
$env:EMBED_MODEL = "mxbai-embed-large"
python bulk_import.py ./docs --reset
```

| Embedding モデル | 次元数 | サイズ | 特徴 |
|---|---|---|---|
| `nomic-embed-text` (デフォルト) | 768 | 274 MB | バランス型。十分な精度 |
| `mxbai-embed-large` | 1024 | 670 MB | より高精度。検索品質が向上 |

> **注意**: VRAM は LLM + Embedding の合計で収まる必要があります。  
> 例: RTX 3070 Ti (8GB) なら `qwen2.5:7b` (5-6GB) + `nomic-embed-text` (0.3GB) ≈ 6GB で OK。

## Caddy でリバースプロキシ（本番向け）

[Caddy](https://caddyserver.com/) をリバースプロキシとして使うと、自動 HTTPS・HTTP/2 が簡単に導入できます。

### インストール

```powershell
winget install CaddyServer.Caddy
```

### 構成

```
クライアント → Caddy (:80 or HTTPS) → Uvicorn (:8000)
```

### ローカル開発

```powershell
# Uvicorn を起動した状態で
caddy run --config Caddyfile
```

http://localhost（ポート 80、`:8000` 不要）でアクセスできます。

### 社内サーバー公開（自動 HTTPS）

`Caddyfile` のコメントアウト部分を編集してドメインを指定:

```caddyfile
rag.internal.example.com {
    reverse_proxy localhost:8000 {
        flush_interval -1
    }
}
```

```powershell
caddy start --config Caddyfile
```

Let's Encrypt による HTTPS 証明書が自動で取得・更新されます。  
`flush_interval -1` によりSSEストリーミングがバッファリングなしで通ります。
