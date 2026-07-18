from __future__ import annotations

import asyncio
import io
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

from app.image_service import ImageService, sanitize_filename
from app.main import create_app
from tests.conftest import FakeLLM, MutableClock, create_conversation


def image_bytes(format_name: str = "PNG", size: tuple[int, int] = (12, 9)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "#2c9a91").save(buffer, format=format_name)
    return buffer.getvalue()


def custom_client(settings: object, clock: MutableClock, fake: FakeLLM) -> Iterator[TestClient]:
    app = create_app(settings, llm_service=fake, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        client.headers.update({"X-Simple-Chat-Request": "1"})
        yield client


def test_multiple_images_and_image_only_message(client: TestClient, fake_llm: FakeLLM) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("two images", "resp_multi")
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": ""},
        files=[
            ("images", ("one.png", image_bytes(), "image/png")),
            ("images", ("two.jpg", image_bytes("JPEG"), "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert len(messages[0]["attachments"]) == 2
    assert len(fake_llm.calls[0]["image_paths"]) == 2


def test_file_count_and_text_length_are_rejected_before_llm(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    too_many = [
        ("images", (f"{index}.png", image_bytes(), "image/png")) for index in range(5)
    ]
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "images"},
        files=too_many,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"
    assert fake_llm.calls == []


def test_per_file_size_limit_returns_413_and_cleans_temp_files(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    settings.images.max_file_size_mb = 1
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        response = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "large"},
            files={"images": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
        )
        assert response.status_code == 413, response.text
        assert response.json()["error"]["code"] == "payload_too_large"
        assert list(settings.tmp_path.iterdir()) == []
        assert list(settings.uploads_path.iterdir()) == []


def test_total_request_size_limit_counts_received_body(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    settings.server.max_request_size_mb = 1
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        response = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "large"},
            files={"images": ("large.png", b"x" * (1024 * 1024 + 128), "image/png")},
        )
        assert response.status_code == 413, response.text
        assert response.json()["error"]["code"] == "payload_too_large"
        assert response.headers["cache-control"] == "no-store"
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert fake_llm.calls == []


def test_corrupt_image_and_mime_spoof_leave_no_files(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    conversation = create_conversation(client)
    corrupt = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "bad"},
        files={"images": ("bad.png", b"not an image", "image/png")},
    )
    assert corrupt.status_code == 400
    assert corrupt.json()["error"]["code"] == "invalid_image"
    spoofed = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "spoof"},
        files={"images": ("fake.jpg", image_bytes("PNG"), "image/jpeg")},
    )
    assert spoofed.status_code == 400
    assert list(settings.tmp_path.iterdir()) == []
    assert list(settings.uploads_path.iterdir()) == []
    assert fake_llm.calls == []


def test_pixel_limit_and_animated_webp_are_rejected(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    settings.images.max_decoded_pixels = 100
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        pixels = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "pixels"},
            files={"images": ("pixels.png", image_bytes(size=(11, 10)), "image/png")},
        )
        assert pixels.status_code == 400

    settings.images.max_decoded_pixels = 40_000_000
    animation = io.BytesIO()
    first = Image.new("RGB", (10, 10), "red")
    second = Image.new("RGB", (10, 10), "blue")
    first.save(
        animation,
        format="WEBP",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        animated = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "animation"},
            files={"images": ("animated.webp", animation.getvalue(), "image/webp")},
        )
        assert animated.status_code == 400
        assert "アニメーション" in animated.json()["error"]["message"]


def test_exif_orientation_resize_and_metadata_removal(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    settings.images.max_dimension_px = 50
    source = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "secret artist"
    Image.new("RGB", (80, 40), "green").save(source, format="JPEG", exif=exif)
    fake_llm.queue_success("ok", "resp_exif")
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        response = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "orientation"},
            files={"images": ("../../camera.jpg", source.getvalue(), "image/jpeg")},
        )
        assert response.status_code == 200
        messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
        attachment = messages[0]["attachments"][0]
        assert attachment["original_name"] == "camera.jpg"
        assert (attachment["width"], attachment["height"]) == (25, 50)
        normalized = client.get(attachment["content_url"])
        with Image.open(io.BytesIO(normalized.content)) as result:
            assert result.format == "WEBP"
            assert result.getexif() == {}
            assert "exif" not in result.info


def test_image_unsupported_model_is_rejected_before_writing(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    settings.llm.enabled_models["gpt-5.6-luna"].supports_images = False
    with next_client(custom_client(settings, clock, fake_llm)) as client:
        conversation = create_conversation(client)
        response = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            data={"text": "image"},
            files={"images": ("image.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "image_not_supported"
        assert fake_llm.calls == []
        assert list(settings.uploads_path.iterdir()) == []


def test_attachment_id_delivery_headers_and_non_public_paths(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("ok", "resp_attachment")
    client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "image"},
        files={"images": ("image.png", image_bytes(), "image/png")},
    )
    attachment = client.get(f"/api/conversations/{conversation['id']}/messages").json()[0][
        "attachments"
    ][0]
    response = client.get(attachment["content_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert client.get("/api/attachments/not-a-uuid/content").status_code == 400
    assert client.get(f"/api/attachments/{uuid.uuid4()}/content").status_code == 404
    assert str(settings.uploads_path) not in str(attachment)
    assert b"data:image" not in settings.database_path.read_bytes()


def test_allowed_origin_no_cors_and_vendor_cache_headers(client: TestClient) -> None:
    allowed = client.post(
        "/api/conversations",
        json={},
        headers={"Origin": "http://localhost:8000", "X-Simple-Chat-Request": "1"},
    )
    assert allowed.status_code == 201
    assert "access-control-allow-origin" not in allowed.headers
    vendor = client.get("/vendor/marked-15.0.12.umd.js")
    assert vendor.status_code == 200
    assert vendor.headers["cache-control"] == "public, max-age=31536000, immutable"
    index = client.get("/")
    assert index.headers["cache-control"] == "no-store"
    assert "unsafe-inline" not in index.headers["content-security-policy"]


def test_symlinked_attachment_is_never_served_or_followed_on_delete(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("ok", "resp_symlink")
    client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "image"},
        files={"images": ("image.png", image_bytes(), "image/png")},
    )
    attachment = client.get(f"/api/conversations/{conversation['id']}/messages").json()[0][
        "attachments"
    ][0]
    with sqlite3.connect(settings.database_path) as connection:
        stored_name = connection.execute("SELECT stored_name FROM attachments").fetchone()[0]
    stored_path = settings.uploads_path / stored_name
    outside = settings.project_root / "outside.webp"
    outside.write_bytes(b"must survive")
    stored_path.unlink()
    try:
        stored_path.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are not available in this Windows environment")

    assert client.get(attachment["content_url"]).status_code == 404
    assert client.delete(f"/api/conversations/{conversation['id']}").status_code == 204
    assert outside.read_bytes() == b"must survive"


@pytest.mark.asyncio
async def test_image_normalization_cancellation_waits_and_cleans_files(
    settings: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ImageService(settings)
    service.initialize()
    original_normalize = service._normalize_image
    started = threading.Event()
    release = threading.Event()

    def blocking_normalize(*args: object) -> tuple[int, int]:
        started.set()
        assert release.wait(timeout=3)
        return original_normalize(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "_normalize_image", blocking_normalize)
    upload = UploadFile(
        file=io.BytesIO(image_bytes()),
        filename="cancel.png",
        headers=Headers({"content-type": "image/png"}),
    )
    task = asyncio.create_task(service.stage([upload]))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(settings.tmp_path.iterdir()) == []
    assert list(settings.uploads_path.iterdir()) == []


def test_filename_sanitizer_never_returns_a_path() -> None:
    assert sanitize_filename("../../secret.png") == "secret.png"
    assert sanitize_filename("..\\..\\secret.png") == "secret.png"
    assert sanitize_filename("a\x00b\x1fc.png") == "abc.png"
    assert len(sanitize_filename("a" * 500 + ".png")) == 200


class next_client:
    """Wrap a generator-based TestClient helper as a context manager."""

    def __init__(self, generator: Iterator[TestClient]) -> None:
        self.generator = generator

    def __enter__(self) -> TestClient:
        return next(self.generator)

    def __exit__(self, *_: object) -> None:
        try:
            next(self.generator)
        except StopIteration:
            pass
