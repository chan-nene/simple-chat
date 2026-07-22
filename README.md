# Simple Chat

OpenAI Responses APIを使う、ローカルPC用チャットアプリです。

## セットアップ（Windows）

Python 3.11以上を使用します。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.toml config.toml
$env:OPENAI_API_KEY = "your-api-key"
python run.py
```

起動後、<http://127.0.0.1:8000>を開きます。APIキーは環境変数にだけ設定してください。

会社PCへは次を同じ構成でコピーします。

```text
run.py
requirements.txt
config.example.toml → config.toml
web/
  index.html
  styles.css
  app.js
  ai-icon.svg
```

## 設定

モデルは`config.toml`を編集してアプリを再起動すると変更できます。

```toml
[model]
target = "APIへ渡すモデルID"
label = "画面に表示するモデル名"
```

AIアイコンは画像を`web/`へ置き、`[ui]`の`ai_icon`へ指定します。同じ画像がfaviconにも使われます。

```toml
[ui]
ai_icon = "/my-ai.png"
```

## 動作

- 更新順で直近5会話を保存
- 正常完了したターンだけを保存
- Response ID失効後の会話は閲覧・削除専用
- 会話DBと添付画像は`data/`へ保存

## CDN

`web/index.html`から直接読み込みます。

| ライブラリ | バージョン | CDN |
|---|---:|---|
| marked | 15.0.12 | jsDelivr |
| DOMPurify | 3.4.11 | jsDelivr |
| highlight.js | 11.11.1 | cdnjs |
| GitHub Dark theme | 11.11.1 | cdnjs |

ライセンスは`THIRD_PARTY_NOTICES.md`を参照してください。

## テスト（任意）

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

このアプリには認証機能がありません。外部公開せず、`127.0.0.1`で使用してください。
