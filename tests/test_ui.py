from __future__ import annotations

import os
import re
import base64
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def ui_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    browser_root = tmp_path_factory.mktemp("browser-app")
    environment = os.environ.copy()
    environment["SIMPLE_CHAT_BROWSER_ROOT"] = str(browser_root)
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.browser_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--workers",
            "1",
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        creationflags=creation_flags,
    )
    try:
        for _ in range(80):
            if process.poll() is not None:
                raise RuntimeError("browser test server exited during startup")
            try:
                if httpx.get(f"{BASE_URL}/api/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("browser test server did not become ready")
        yield BASE_URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


def test_desktop_chat_stream_markdown_model_boundary_and_sanitization(
    browser: Browser, ui_server: str
) -> None:
    page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
    console_errors: list[str] = []
    unexpected_dialogs: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)

    def dismiss_unexpected(dialog: object) -> None:
        unexpected_dialogs.append(dialog.message)
        dialog.dismiss()

    page.on("dialog", dismiss_unexpected)
    page.goto(ui_server, wait_until="networkidle")
    expect(page.get_by_role("heading", name="Simple Chat")).to_be_visible()
    expect(page.get_by_role("textbox", name="メッセージ")).to_be_enabled()
    expect(page.get_by_text("ここから会話を始めます")).to_be_visible()
    expect(page.locator(".topbar")).to_have_count(0)
    expect(page.locator("#sidebar #model-select")).to_be_visible()
    expect(page.locator("#retention-days")).to_have_text("7日")

    rename_button = page.locator(".conversation-item.active").get_by_role(
        "button", name="会話タイトルを変更"
    )
    rename_button.focus()
    rename_button.click()
    expect(page.get_by_role("dialog", name="会話タイトルを変更")).to_be_visible()
    page.locator("#rename-cancel").click()
    expect(rename_button).to_be_focused()

    page.get_by_role("textbox", name="メッセージ").fill("Responses API の連鎖を確認")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator(".message.assistant .code-frame")).to_be_visible(timeout=12_000)
    copy_button = page.get_by_role("button", name="コードをコピー")
    expect(copy_button).to_be_visible()
    expect(copy_button.locator("svg.copy-icon")).to_be_visible()
    expect(page.locator("#send-button svg")).to_have_count(1)
    expect(page.locator(".message.assistant")).to_contain_text("Response ID で文脈を継続")
    message_layout = page.evaluate(
        """() => {
          const assistant = document.querySelector('.message.assistant').getBoundingClientRect();
          const user = document.querySelector('.message.user').getBoundingClientRect();
          const avatar = document.querySelector('.message.user .user-avatar').getBoundingClientRect();
          const assistantBody = document.querySelector('.message.assistant .message-body');
          const active = document.querySelector('.conversation-item.active');
          return {
            assistantLeft: assistant.left,
            userLeft: user.left,
            avatarLeft: avatar.left,
            userRight: user.right,
            bodyFontSize: getComputedStyle(assistantBody).fontSize,
            activeBackground: getComputedStyle(active).backgroundColor,
            activeBeforeContent: getComputedStyle(active, '::before').content,
          };
        }"""
    )
    assert message_layout["assistantLeft"] < message_layout["userLeft"]
    assert message_layout["avatarLeft"] < message_layout["userRight"]
    assert message_layout["bodyFontSize"] == "16px"
    assert message_layout["activeBackground"] == "rgb(34, 48, 60)"
    assert message_layout["activeBeforeContent"] in {"none", "normal"}

    sidebar_toggle = page.get_by_role("button", name="サイドバーを折りたたむ")
    sidebar_toggle.click()
    expect(page.locator("#app-shell")).to_have_class(re.compile(r".*sidebar-collapsed.*"))
    expect(page.locator("#sidebar")).to_have_css("width", "58px")
    page.get_by_role("button", name="サイドバーを展開する").click()
    expect(page.locator("#app-shell")).not_to_have_class(re.compile(r".*sidebar-collapsed.*"))

    page.get_by_label("使用モデル").focus()
    page.get_by_label("使用モデル").select_option("gpt-5.6-terra")
    expect(page.get_by_role("dialog", name="モデルを変更しますか？")).to_be_visible()
    page.locator("#dialog-confirm").click()
    expect(page.get_by_label("使用モデル")).to_be_focused()
    expect(page.locator(".message.user")).to_have_count(1)
    expect(page.locator(".context-boundary")).to_have_count(1)
    expect(page.locator(".context-boundary")).to_have_text(
        "モデルをGPT-5.6 Terraに変更しました。ここから新しい文脈です"
    )

    page.get_by_role("textbox", name="メッセージ").fill(
        '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==" onerror="alert(1)">'
    )
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator(".message.assistant")).to_have_count(2, timeout=12_000)
    expect(page.locator(".message.assistant").last.locator(".status-badge")).to_have_count(
        0, timeout=12_000
    )
    expect(page.locator(".message.assistant [onerror]")).to_have_count(0)
    expect(page.locator(".context-boundary")).to_have_count(1)

    output = PROJECT_ROOT / "test-results" / "ui-desktop.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=output, full_page=True)
    assert unexpected_dialogs == []
    assert console_errors == []
    page.close()


def test_mobile_sidebar_and_composer_are_usable(browser: Browser, ui_server: str) -> None:
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    page.goto(ui_server, wait_until="networkidle")
    page.get_by_role("button", name="サイドバーを開く").click()
    expect(page.locator("#app-shell")).to_have_class(re.compile(r".*sidebar-open.*"))
    expect(page.get_by_role("button", name="新しいチャット", exact=True)).to_be_visible()
    page.get_by_role("button", name="サイドバーを閉じる").click()
    expect(page.locator("#app-shell")).not_to_have_class(re.compile(r".*sidebar-open.*"))
    page.wait_for_timeout(250)
    expect(page.get_by_role("textbox", name="メッセージ")).to_be_visible()
    expect(page.get_by_role("button", name="画像を添付")).to_be_visible()
    expect(page.locator("#upload-limits")).to_contain_text("全体45MB")
    composer_box = page.locator("#composer-zone").bounding_box()
    layout = page.evaluate(
        """() => Object.fromEntries(['app-shell', 'message-stage', 'composer-zone'].map(id => {
          const node = document.getElementById(id);
          const rect = node.getBoundingClientRect();
          return [id, {top: rect.top, bottom: rect.bottom, height: rect.height, overflow: getComputedStyle(node).overflow}];
        }).concat([['main-panel', (() => {
          const node = document.querySelector('.main-panel');
          const rect = node.getBoundingClientRect();
          return {top: rect.top, bottom: rect.bottom, height: rect.height, rows: getComputedStyle(node).gridTemplateRows};
        })()]]))"""
    )
    assert composer_box is not None
    assert composer_box["y"] >= 0
    assert composer_box["y"] + composer_box["height"] <= 844, layout
    output = PROJECT_ROOT / "test-results" / "ui-mobile.png"
    page.screenshot(path=output, full_page=True)
    page.close()


def test_stop_and_ime_enter_reconcile_to_persisted_state(browser: Browser, ui_server: str) -> None:
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    page.goto(ui_server, wait_until="networkidle")
    page.get_by_role("button", name="新しいチャット", exact=True).click()
    editor = page.get_by_role("textbox", name="メッセージ")
    editor.fill("__slow__")
    editor.press("Enter")
    expect(page.locator(".message.assistant .status-badge")).to_have_text("生成中")
    page.get_by_role("button", name="停止").click()
    expect(page.locator(".message.assistant .status-badge")).to_have_text("停止", timeout=12_000)
    expect(page.locator(".excluded-badge")).to_have_count(2)
    expect(page.locator(".context-boundary")).to_have_count(0)

    user_count = page.locator(".message.user").count()
    editor.fill("IMEで確定")
    editor.dispatch_event("compositionstart")
    editor.dispatch_event("keydown", {"key": "Enter", "isComposing": True})
    page.wait_for_timeout(250)
    expect(page.locator(".message.user")).to_have_count(user_count)
    editor.dispatch_event("compositionend")
    editor.dispatch_event("keydown", {"key": "Enter", "isComposing": False})
    expect(page.locator(".message.user")).to_have_count(user_count + 1, timeout=12_000)
    expect(page.locator(".message.assistant").last.locator(".status-badge")).to_have_count(
        0, timeout=12_000
    )
    page.close()


def test_file_picker_paste_and_drop_images(browser: Browser, ui_server: str) -> None:
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    page.goto(ui_server, wait_until="networkidle")
    page.get_by_role("button", name="新しいチャット", exact=True).click()
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    payload = {
        "name": "picker.png",
        "mimeType": "image/png",
        "buffer": base64.b64decode(png_base64),
    }
    page.locator("#file-input").set_input_files(payload)
    expect(page.locator(".attachment-preview")).to_have_count(1)
    page.locator(".remove-attachment").click()
    expect(page.locator(".attachment-preview")).to_have_count(0)

    page.locator("#composer-input").evaluate(
        """(element, encoded) => {
          const bytes = Uint8Array.from(atob(encoded), value => value.charCodeAt(0));
          const transfer = new DataTransfer();
          transfer.items.add(new File([bytes], 'pasted.png', {type: 'image/png'}));
          element.dispatchEvent(new ClipboardEvent('paste', {clipboardData: transfer, bubbles: true, cancelable: true}));
        }""",
        png_base64,
    )
    expect(page.locator(".attachment-preview")).to_have_count(1)
    page.locator("#composer-zone").evaluate(
        """(element, encoded) => {
          const bytes = Uint8Array.from(atob(encoded), value => value.charCodeAt(0));
          const transfer = new DataTransfer();
          transfer.items.add(new File([bytes], 'dropped.png', {type: 'image/png'}));
          element.dispatchEvent(new DragEvent('dragenter', {dataTransfer: transfer, bubbles: true, cancelable: true}));
          element.dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true}));
        }""",
        png_base64,
    )
    expect(page.locator(".attachment-preview")).to_have_count(2)
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator(".message.user .message-images img")).to_have_count(2, timeout=12_000)
    expect(page.locator(".message.assistant .status-badge")).to_have_count(0, timeout=12_000)
    page.close()


def test_scrolling_up_pauses_auto_follow(browser: Browser, ui_server: str) -> None:
    page = browser.new_page(viewport={"width": 1100, "height": 680})
    page.goto(ui_server, wait_until="networkidle")
    page.get_by_role("button", name="新しいチャット", exact=True).click()
    editor = page.get_by_role("textbox", name="メッセージ")
    editor.fill("__long__")
    editor.press("Enter")
    expect(page.locator(".message.assistant .status-badge")).to_have_text("生成中")
    for _ in range(60):
        overflow = page.locator("#message-stage").evaluate(
            "element => element.scrollHeight - element.clientHeight"
        )
        if overflow > 180:
            break
        page.wait_for_timeout(50)
    assert overflow > 180
    page.locator("#message-stage").evaluate("element => { element.scrollTop = 0; }")
    expect(page.get_by_role("button", name=re.compile("最新へ移動"))).to_be_visible(timeout=12_000)
    scroll_top = page.locator("#message-stage").evaluate("element => element.scrollTop")
    assert scroll_top < 100
    page.get_by_role("button", name=re.compile("最新へ移動")).click()
    page.wait_for_timeout(200)
    distance = page.locator("#message-stage").evaluate(
        "element => element.scrollHeight - element.scrollTop - element.clientHeight"
    )
    assert distance < 100
    page.get_by_role("button", name="停止").click()
    page.close()
