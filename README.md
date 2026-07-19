# Simple Chat

OpenAI Responses APIを利用する、ローカル実行向けのシンプルなチャットアプリです。FastAPIとVanilla JavaScriptで構成され、会話履歴はローカルのSQLiteに保存します。

## 主な機能

- Responses APIによるストリーミング応答
- 会話の作成、タイトル変更、削除
- Markdown、コードブロック表示
- JPEG、PNG、WebP画像の添付
- モデル切り替え
- 一定期間を過ぎた会話と添付ファイルの自動削除
- APIキー未設定時の閲覧専用モード

## 必要環境

- Python 3.11以上
- OpenAI APIキー

## セットアップ

リポジトリを取得し、仮想環境を作成して依存関係をインストールします。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.toml config.toml
$env:OPENAI_API_KEY = "your-api-key"
python run.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.toml config.toml
export OPENAI_API_KEY="your-api-key"
python run.py
```

起動後、<http://127.0.0.1:8000> をブラウザで開いてください。

> [!IMPORTANT]
> APIキーは環境変数で設定し、設定ファイルやソースコードへ書き込まないでください。

## 設定

`config.example.toml` を `config.toml` にコピーして編集します。`config.toml` はGitの管理対象外です。

主に次の項目を変更できます。

- 使用するモデル
- 応答のinstructionsと最大トークン数
- 画像のサイズ、枚数、形式
- 会話履歴の保持日数
- ポート番号と画面タイトル
- AIメッセージのアイコン画像

AIアイコンは画像を `app/static/` 配下へ置き、`[ui]` の `ai_icon` にルート相対パスを指定します。
たとえば `app/static/my-ai.png` を使う場合は `ai_icon = "/my-ai.png"` と記載します。

別の場所にある設定ファイルを使う場合は、`SIMPLE_CHAT_CONFIG` 環境変数にパスを指定してください。

APIキーを設定せずに起動した場合、保存済みの会話は閲覧できますが、新しいメッセージは送信できません。

## データ保存とプライバシー

ローカルデータは既定で `data/` 配下に保存されます。このディレクトリはGitの管理対象外です。

- 会話履歴: `data/chat.db`
- 添付画像: `data/uploads/`
- 一時ファイル: `data/tmp/`

会話の文脈にはOpenAIの `previous_response_id` を使用します。メッセージ本文や画像は、応答生成のためOpenAI APIへ送信されます。利用前にOpenAIのデータ利用条件を確認してください。

このアプリには認証機能がありません。サーバーは `127.0.0.1` でのみ起動する設計であり、LANやインターネットへ公開しないでください。

## テスト

```bash
python -m pip install -r requirements-dev.txt
playwright install chromium
python -m pytest -q
```

通常のテストはOpenAI APIへ接続しません。

実APIとの疎通確認は、APIキーを設定したうえで明示的に有効化します。

```bash
SIMPLE_CHAT_RUN_REAL_API=1 python tools/smoke_openai.py
```

PowerShellでは次のように実行します。

```powershell
$env:SIMPLE_CHAT_RUN_REAL_API = "1"
python tools\smoke_openai.py
```

## 補足

- ブラウザ向けライブラリは `app/static/vendor/` に同梱しています。
- 本アプリ自体のライセンスｈが現状未指定です。
- 各ライブラリのライセンス文は同ディレクトリにあります。
- 詳細な設定例は [config.example.toml](config.example.toml) を参照してください。
