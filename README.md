# 社内RAGアシスタント

Ollama + ChromaDB を使った完全ローカルの RAG（検索拡張生成）ウェブアプリです。  
社内ドキュメントをアップロードし、自然言語で質問すると、ドキュメントに基づいた回答が得られます。

**社外にデータを送信しません。** すべてローカルで完結します。

## 特徴

- 📄 **マルチフォーマット対応** — PDF / Word (.docx) / Excel (.xlsx) / テキスト (.txt)
- 🔍 **OCR対応** — スキャン画像PDFからもテキスト抽出（Tesseract jpn+eng）
- 🎯 **3つの回答モード** — RAGモード / ハイブリッド / フリーチャット
- 🤖 **モデル切り替え** — UIからOllamaのモデルを選択可能
- ⚡ **ストリーミング応答** — リアルタイムに回答を表示（SSE）
- 💬 **チャット履歴永続化** — SQLiteでセッション管理、ブラウザを閉じても復元可能
- 🔒 **マルチユーザー認証** — Cookie認証、管理者/一般ユーザーの権限分離
- 👥 **ユーザー管理** — 管理画面からユーザー作成・削除・パスワードリセット
- 🌙 **ダークモード** — ワンクリックで切り替え、設定はブラウザに保存
- 📦 **Docker対応** — `docker compose up` で一発起動
- 🧪 **CI (GitHub Actions)** — push / PR 時にテスト自動実行
- 🖥️ **クロスプラットフォーム** — Windows / Linux / macOS で動作

## アーキテクチャ

```
[ドキュメント登録（管理者）]
ファイル → テキスト抽出(OCR対応) → チャンク分割 → Embedding(nomic-embed-text) → ChromaDB

[質問回答（ユーザー）]
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
| チャット履歴 | SQLite |
| バックエンド | FastAPI (Python) |
| フロントエンド | HTML / CSS / JavaScript |
| OCR | Tesseract (jpn+eng) |

---

## クイックスタート

### 方法 A: Docker（推奨）

```bash
git clone https://github.com/ebians/ollama.git
cd ollama
docker compose up -d --build
docker exec ollama ollama pull qwen2.5:7b
docker exec ollama ollama pull nomic-embed-text
```

→ http://localhost:8000 にアクセス

### 方法 B: ローカル実行

```bash
git clone https://github.com/ebians/ollama.git
cd ollama

# Ollama モデル取得
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Python 環境構築
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# サーバー起動
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

→ http://localhost:8000 にアクセス

---

## 必要環境

| 項目 | 要件 |
|---|---|
| **Python** | 3.11 以上 |
| **Ollama** | [https://ollama.com](https://ollama.com) |
| **GPU** | NVIDIA GPU (8GB VRAM 以上) 推奨。CPU のみでも動作可 |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| **Docker** (方法Aの場合) | Docker Engine 20.10+ または Docker Desktop |

---

## セットアップ詳細

### Ollama のインストール

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
# または https://ollama.com からダウンロード
```

**Windows:**
https://ollama.com からインストーラをダウンロード

### モデルの取得

```bash
ollama pull qwen2.5:7b        # チャットモデル (4.7GB)
ollama pull nomic-embed-text   # 埋め込みモデル (274MB)
```

> モデルの保存先を変更したい場合は、環境変数 `OLLAMA_MODELS` にパスを設定してください。  
> 例: `export OLLAMA_MODELS=/mnt/data/ollama` (Linux) / `$env:OLLAMA_MODELS="D:\llama"` (Windows)

### Python 環境構築

**Linux / macOS:**
```bash
cd ollama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
cd ollama
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Tesseract OCR（スキャンPDF対応、任意）

画像PDFのOCR抽出を使わない場合はスキップ可能です。

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
```

**Linux (RHEL/Fedora):**
```bash
sudo dnf install tesseract tesseract-langpack-jpn tesseract-langpack-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
[UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) からインストーラをダウンロード。  
インストール先を `TESSERACT_CMD` 環境変数で指定:
```powershell
$env:TESSERACT_CMD = "D:\tools\Tesseract\tesseract.exe"
```

> Docker を使う場合は Tesseract が自動でインストールされるため設定不要です。

### サーバー起動

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

ブラウザで **http://localhost:8000** にアクセスしてください。

---

## 使い方

### 1. ドキュメント登録（管理者）

http://localhost:8000/admin にアクセスし、管理者アカウントでログイン（初回: ユーザー名 `admin` / パスワード `admin`）。

- ファイルをアップロード（対応形式: `.txt` / `.pdf` / `.docx` / `.xlsx`）
- 一覧から個別削除 / 全削除が可能

### 2. 質問する（ユーザー）

http://localhost:8000 にアクセス。

- **RAGモード**: ドキュメントのみに基づいて回答
- **ハイブリッドモード**: ドキュメント + LLMの知識で回答
- **フリーチャット**: ドキュメント参照なし、LLMの知識のみで回答

回答はストリーミングでリアルタイム表示されます。  
サイドバーからチャット履歴の確認・復元・削除が可能です。

### 3. フォルダ一括登録（bulk_import.py）

大量のドキュメントをCLIから一括登録する場合に使います。

```bash
# 基本: フォルダを指定して一括登録（サブフォルダも再帰探索）
python bulk_import.py ./docs

# DB を初期化してから登録（やり直したい場合）
python bulk_import.py ./docs --reset

# 追加分だけ登録
python bulk_import.py /path/to/new_docs
```

---

## API エンドポイント

### 認証

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/auth/login` | ログイン（Cookie 発行） |
| `POST` | `/api/auth/logout` | ログアウト |
| `GET` | `/api/auth/me` | ログイン中のユーザー情報 |
| `POST` | `/api/auth/password` | パスワード変更（自分自身） |

### ユーザー向け（要ログイン）

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/ask` | 質問（一括回答） |
| `POST` | `/api/ask/stream` | 質問（SSE ストリーミング） |
| `GET` | `/api/stats` | DB 統計 |
| `GET` | `/api/models` | 利用可能なモデル一覧 |
| `GET` | `/api/sessions` | チャットセッション一覧 |
| `GET` | `/api/sessions/{id}` | セッションのメッセージ取得 |
| `DELETE` | `/api/sessions/{id}` | セッション削除 |

### 管理者向け（要管理者ログイン）

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/upload` | ファイルアップロード |
| `GET` | `/api/documents` | 登録ドキュメント一覧 |
| `GET` | `/api/documents/{source}/chunks` | チャンクプレビュー |
| `DELETE` | `/api/documents/{source}` | ドキュメント削除 |
| `POST` | `/api/reset` | 全ドキュメント削除 |
| `GET` | `/api/users` | ユーザー一覧 |
| `POST` | `/api/users` | ユーザー作成 |
| `DELETE` | `/api/users/{id}` | ユーザー削除 |
| `POST` | `/api/users/{id}/reset-password` | パスワードリセット |

### リクエスト例

```bash
# ログイン（Cookie 取得）
curl -c cookies.txt -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# 質問（ストリーミング）
curl -b cookies.txt -N -X POST http://localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "有給休暇の取得方法は？", "mode": "rag"}'

# ファイルアップロード（管理者）
curl -b cookies.txt -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"

# セッション一覧
curl -b cookies.txt http://localhost:8000/api/sessions
```

---

## 設定（環境変数）

| 環境変数 | デフォルト値 | 説明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama サーバーの URL |
| `CHAT_MODEL` | `qwen2.5:7b` | チャットに使用するモデル |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding に使用するモデル |
| `CHROMA_DIR` | `./chroma_data` | ChromaDB のデータ保存先 |
| `CHAT_DB_PATH` | `./chat_history.db` | チャット履歴 DB のパス |
| `CHUNK_SIZE` | `500` | チャンク分割の目安文字数 |
| `CHUNK_OVERLAP` | `50` | チャンク間のオーバーラップ文字数 |
| `TOP_K` | `3` | 検索で取得するチャンク数 |
| `ADMIN_PASSWORD` | `admin` | 初期管理者パスワード（初回起動時に admin ユーザーを自動作成） |
| `TESSERACT_CMD` | (OS 依存) | Tesseract 実行ファイルのパス |

Linux の設定例:
```bash
export ADMIN_PASSWORD="your-secret"
export OLLAMA_BASE_URL="http://192.168.1.100:11434"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows の設定例:
```powershell
$env:ADMIN_PASSWORD = "your-secret"
$env:OLLAMA_BASE_URL = "http://192.168.1.100:11434"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## プロジェクト構成

```
ollama/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI（テスト自動実行）
├── app/
│   ├── __init__.py
│   ├── config.py            # 設定（環境変数から読み込み）
│   ├── chat_store.py        # チャット履歴 (SQLite)
│   ├── main.py              # FastAPI エンドポイント
│   ├── ollama_client.py     # Ollama API クライアント（Embedding / Chat / Streaming）
│   ├── parser.py            # ファイル解析（txt / pdf / docx / xlsx / OCR）
│   ├── user_store.py        # ユーザー管理・認証 (SQLite + bcrypt)
│   └── vectorstore.py       # ChromaDB 操作（登録 / 検索 / リセット）
├── tests/
│   ├── test_api.py          # API エンドポイントテスト
│   ├── test_chat_store.py   # チャット履歴テスト
│   ├── test_parser.py       # ファイル解析テスト
│   └── test_vectorstore.py  # ベクトルDB テスト
├── static/
│   ├── index.html           # ユーザー向け UI（ダークモード対応）
│   └── admin.html           # 管理者向け UI（ダークモード対応）
├── docs/                    # サンプルドキュメント
├── bulk_import.py           # フォルダ一括登録スクリプト
├── Caddyfile                # Caddy リバースプロキシ設定
├── Dockerfile               # アプリコンテナ定義
├── docker-compose.yml       # 一発起動定義
├── .dockerignore
├── requirements.txt
└── .gitignore
```

---

## Docker で起動

Docker Compose を使うと、Ollama + アプリを一発で起動できます。

### 前提条件

- Docker Engine 20.10+ または [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- NVIDIA GPU を使う場合: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Linux に Docker をインストール:**
```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit（GPU 使用時）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 起動

```bash
docker compose up -d --build
```

初回起動時はモデルのダウンロードが必要です:

```bash
docker exec ollama ollama pull qwen2.5:7b
docker exec ollama ollama pull nomic-embed-text
```

ブラウザで **http://localhost:8000** にアクセスしてください。

### コード更新後のコンテナ更新

コードを変更した後、Docker コンテナに反映するには:

```bash
# イメージを再ビルドしてコンテナを再作成
docker compose up -d --build

# app コンテナだけ更新する場合
docker compose up -d --build app

# キャッシュなしで完全再ビルド（依存パッケージが変わった場合など）
docker compose build --no-cache app
docker compose up -d
```

`app_data` / `ollama_models` ボリュームのデータはそのまま保持されます。

Ollama イメージを最新に更新する場合:

```bash
docker compose pull ollama
docker compose up -d
```

不要な旧イメージの掃除:

```bash
docker image prune -f
```

### 停止

```bash
docker compose down
```

### データの永続化

| ボリューム | 内容 |
|---|---|
| `ollama_models` | Ollama のモデルデータ |
| `app_data` | ChromaDB + チャット履歴 (SQLite) |

ボリュームは `docker compose down` では削除されません。  
完全に削除する場合は `docker compose down -v` を使います。

### 環境変数のカスタマイズ

`docker-compose.yml` の `environment` セクションで設定を変更できます:

```yaml
environment:
  - CHAT_MODEL=qwen2.5:14b      # モデル変更
  - ADMIN_PASSWORD=your-secret   # 管理者パスワード変更
  - TOP_K=5                      # 検索チャンク数変更
```

### GPU なしで起動（CPU のみ）

`docker-compose.yml` の `ollama` サービスから `deploy` セクションを削除してください:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    # deploy セクションを削除
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
```

> **注意**: CPU モードでは応答速度が大幅に低下します。

### ホストのモデルデータを共有する

ホストに既にダウンロード済みのモデルを Docker コンテナで共有できます:

```yaml
# docker-compose.yml
services:
  ollama:
    volumes:
      # Linux
      - /usr/share/ollama/.ollama:/root/.ollama
      # Windows
      # - D:/llama:/root/.ollama
```

---

## LLM の差し替え

環境変数 `CHAT_MODEL` を変更するだけで、コード修正なしにモデルを差し替えられます。

```bash
# Linux / macOS
ollama pull qwen2.5:14b
export CHAT_MODEL="qwen2.5:14b"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Windows
ollama pull qwen2.5:14b
$env:CHAT_MODEL = "qwen2.5:14b"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker exec ollama ollama pull qwen2.5:14b
# docker-compose.yml の CHAT_MODEL を変更して再起動
docker compose up -d
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

### Embedding モデルの差し替え

Embedding モデルも同様に差し替え可能です。ただし **変更後は DB リセット＋再登録が必要** です（ベクトルの次元数が異なるため）。

```bash
ollama pull mxbai-embed-large
export EMBED_MODEL="mxbai-embed-large"
python bulk_import.py ./docs --reset
```

| Embedding モデル | 次元数 | サイズ | 特徴 |
|---|---|---|---|
| `nomic-embed-text` (デフォルト) | 768 | 274 MB | バランス型。十分な精度 |
| `mxbai-embed-large` | 1024 | 670 MB | より高精度。検索品質が向上 |

> **注意**: VRAM は LLM + Embedding の合計で収まる必要があります。  
> 例: RTX 3070 Ti (8GB) なら `qwen2.5:7b` (5-6GB) + `nomic-embed-text` (0.3GB) ≈ 6GB で OK。

---

## GPU メモリ使用量（参考）

RTX 3070 Ti (8GB VRAM) での実測:

| モデル | サイズ | VRAM 使用量 |
|---|---|---|
| qwen2.5:7b | 4.7 GB | ~5-6 GB |
| nomic-embed-text | 274 MB | ~0.3 GB |
| **合計** | | **~6 GB** |

8GB VRAM に余裕で収まります。

---

## Caddy でリバースプロキシ（本番向け）

[Caddy](https://caddyserver.com/) をリバースプロキシとして使うと、自動 HTTPS・HTTP/2 が簡単に導入できます。

### インストール

```bash
# Linux (Ubuntu/Debian)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy

# macOS
brew install caddy

# Windows
winget install CaddyServer.Caddy
```

### 構成

```
クライアント → Caddy (:80 or HTTPS) → Uvicorn (:8000)
```

### ローカル開発

```bash
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

```bash
caddy start --config Caddyfile
```

Let's Encrypt による HTTPS 証明書が自動で取得・更新されます。  
`flush_interval -1` によりSSEストリーミングがバッファリングなしで通ります。

---

## systemd で自動起動（Linux サーバー）

Linux サーバーで常時稼働させる場合の設定例です。

### Ollama（通常はインストール時に自動設定済み）

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

### RAGアプリ

```bash
sudo tee /etc/systemd/system/rag-app.service << 'EOF'
[Unit]
Description=RAG Assistant App
After=network.target ollama.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/opt/ollama
Environment="PATH=/opt/ollama/.venv/bin:/usr/bin"
Environment="ADMIN_PASSWORD=your-secret"
ExecStart=/opt/ollama/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-app
sudo systemctl start rag-app
```

### ログ確認

```bash
journalctl -u rag-app -f
```

---

## テスト

```bash
# 全テスト実行
pip install pytest pytest-asyncio
pytest tests/ -v

# 特定のテストファイルのみ
pytest tests/test_api.py -v
```

GitHub に push すると、[GitHub Actions](.github/workflows/ci.yml) で自動的にテストが実行されます。

---

## トラブルシューティング

| 問題 | 対処 |
|---|---|
| `Connection refused` (Ollama) | `ollama serve` が起動しているか確認。Docker の場合は `docker logs ollama` |
| ドキュメントが見つからない | 管理画面 (/admin) からアップロード済みか確認。Docker は別ボリュームなので再登録が必要 |
| OCRが動かない | `tesseract --version` で Tesseract がインストール済みか確認 |
| CUDA out of memory | より小さいモデル (`gemma3:4b` 等) に切り替えるか、他のGPU使用アプリを終了 |
| ChromaDB telemetry エラー | 無害な警告です。動作に影響しません |
| Docker で GPU が使えない | `nvidia-smi` と `nvidia-container-toolkit` がインストール済みか確認 |
| ポート 8000 が使用中 | `lsof -i :8000` (Linux) / `netstat -ano \| findstr :8000` (Windows) で確認して停止 |

---

## ライセンス

MIT
