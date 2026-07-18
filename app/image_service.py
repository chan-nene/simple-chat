from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .domain import StagedAttachment
from .errors import AppError


_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_ORPHAN_NAME = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}(?:\.webp|\.upload)?$",
    re.IGNORECASE,
)
_T = TypeVar("_T")


class ImageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.uploads_dir = settings.uploads_path
        self.tmp_dir = settings.tmp_path

    def initialize(self) -> None:
        self._assert_no_symlink_components(
            self.settings.project_root / self.settings.storage.upload_directory
        )
        self._assert_no_symlink_components(
            self.settings.project_root / self.settings.storage.temp_directory
        )
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._assert_safe_directory(self.uploads_dir)
        self._assert_safe_directory(self.tmp_dir)
        Image.MAX_IMAGE_PIXELS = self.settings.images.max_decoded_pixels

    async def stage(self, files: list[UploadFile]) -> list[StagedAttachment]:
        if not files:
            return []
        if not self.settings.images.enabled:
            raise AppError("invalid_image", "画像添付は無効です。", 400, False)
        if len(files) > self.settings.images.max_files:
            raise AppError(
                "invalid_image",
                f"画像は{self.settings.images.max_files}枚まで添付できます。",
                400,
                False,
            )

        staged: list[StagedAttachment] = []
        try:
            for upload in files:
                staged.append(await self._stage_one(upload))
            return staged
        except BaseException:
            self.remove_staged(staged)
            raise
        finally:
            for upload in files:
                await upload.close()

    async def _stage_one(self, upload: UploadFile) -> StagedAttachment:
        claimed_mime = (upload.content_type or "").lower()
        if claimed_mime not in self.settings.images.allowed_types:
            raise AppError("invalid_image", "対応していない画像形式です。", 400, False)

        identifier = str(uuid.uuid4())
        temp_path = self._safe_child(self.tmp_dir, f"{identifier}.upload")
        normalized_path = self._safe_child(self.tmp_dir, f"{identifier}.webp")
        final_name = f"{identifier}.webp"
        final_path = self._safe_child(self.uploads_dir, final_name)
        source_byte_size = 0

        try:
            with temp_path.open("xb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    source_byte_size += len(chunk)
                    if source_byte_size > self.settings.images.max_file_bytes:
                        raise AppError(
                            "payload_too_large",
                            "画像ファイルの容量が上限を超えています。",
                            413,
                            False,
                        )
                    destination.write(chunk)

            if source_byte_size == 0:
                raise AppError("invalid_image", "空の画像ファイルは送信できません。", 400, False)

            width, height = await _protected_to_thread(
                self._normalize_image, temp_path, normalized_path, claimed_mime
            )
            await _protected_to_thread(os.replace, normalized_path, final_path)

            byte_size = final_path.stat().st_size
            digest = await _protected_to_thread(_sha256_path, final_path)
            return StagedAttachment(
                id=identifier,
                original_name=sanitize_filename(upload.filename),
                stored_name=final_name,
                original_mime_type=claimed_mime,
                stored_mime_type="image/webp",
                width=width,
                height=height,
                source_byte_size=source_byte_size,
                byte_size=byte_size,
                sha256=digest,
                path=final_path,
            )
        except AppError:
            _unlink_quietly(final_path)
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            _unlink_quietly(final_path)
            raise AppError("invalid_image", "画像を安全に読み取れませんでした。", 400, False) from None
        except BaseException:
            _unlink_quietly(final_path)
            raise
        finally:
            _unlink_quietly(temp_path)
            _unlink_quietly(normalized_path)

    def _normalize_image(
        self, temp_path: Path, final_path: Path, claimed_mime: str
    ) -> tuple[int, int]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(temp_path) as source:
                actual_mime = _MIME_BY_FORMAT.get(source.format or "")
                if actual_mime is None or actual_mime not in self.settings.images.allowed_types:
                    raise AppError("invalid_image", "対応していない画像形式です。", 400, False)
                if actual_mime != claimed_mime:
                    raise AppError(
                        "invalid_image",
                        "申告された形式と画像の実形式が一致しません。",
                        400,
                        False,
                    )
                if bool(getattr(source, "is_animated", False)) or int(
                    getattr(source, "n_frames", 1)
                ) > 1:
                    raise AppError(
                        "invalid_image", "アニメーション画像には対応していません。", 400, False
                    )
                if source.width * source.height > self.settings.images.max_decoded_pixels:
                    raise AppError("invalid_image", "画像の総画素数が上限を超えています。", 400, False)

                source.load()
                normalized = ImageOps.exif_transpose(source)
                normalized.thumbnail(
                    (
                        self.settings.images.max_dimension_px,
                        self.settings.images.max_dimension_px,
                    ),
                    Image.Resampling.LANCZOS,
                )
                if "A" in normalized.getbands():
                    output = normalized.convert("RGBA")
                else:
                    output = normalized.convert("RGB")
                try:
                    with final_path.open("xb") as destination:
                        output.save(
                            destination,
                            format="WEBP",
                            quality=self.settings.images.webp_quality,
                            method=6,
                        )
                    return output.size
                finally:
                    output.close()

    def remove_staged(self, staged: list[StagedAttachment]) -> None:
        for attachment in staged:
            try:
                safe_path = self._safe_child(self.uploads_dir, attachment.stored_name)
                safe_path.unlink(missing_ok=True)
            except OSError:
                pass

    def delete_path(self, path: Path) -> None:
        safe_path = self._safe_child(self.uploads_dir, path.name)
        if safe_path == path.resolve(strict=False):
            safe_path.unlink(missing_ok=True)

    def cleanup_orphans(self, referenced: set[str], older_than_timestamp: float) -> int:
        removed = 0
        for directory in (self.uploads_dir, self.tmp_dir):
            self._assert_safe_directory(directory)
            for path in directory.iterdir():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or not _ORPHAN_NAME.fullmatch(path.name)
                    or path.name in referenced
                ):
                    continue
                try:
                    if path.stat().st_mtime < older_than_timestamp:
                        self._safe_child(directory, path.name).unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    continue
        return removed

    @staticmethod
    def _assert_safe_directory(directory: Path) -> None:
        if directory.is_symlink():
            raise RuntimeError(f"storage directory must not be a symbolic link: {directory}")
        if not directory.is_dir():
            raise RuntimeError(f"storage directory is unavailable: {directory}")

    def _assert_no_symlink_components(self, declared: Path) -> None:
        root = self.settings.project_root.absolute()
        candidate = declared.absolute()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("storage directory leaves the project root") from exc
        current = root
        if current.exists() and current.is_symlink():
            raise RuntimeError(f"storage path contains a symbolic link: {current}")
        for part in relative.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RuntimeError(f"storage path contains a symbolic link: {current}")

    @staticmethod
    def _safe_child(directory: Path, name: str) -> Path:
        if Path(name).name != name:
            raise RuntimeError("unsafe storage name")
        resolved_directory = directory.resolve(strict=True)
        declared_candidate = resolved_directory / name
        if declared_candidate.is_symlink():
            raise RuntimeError("storage file must not be a symbolic link")
        candidate = declared_candidate.resolve(strict=False)
        try:
            candidate.relative_to(resolved_directory)
        except ValueError as exc:
            raise RuntimeError("storage path leaves configured directory") from exc
        return candidate


def sanitize_filename(value: str | None) -> str:
    name = Path((value or "image").replace("\\", "/")).name
    cleaned = _CONTROL_CHARS.sub("", name).strip()
    return (cleaned or "image")[:200]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _protected_to_thread(function: Callable[..., _T], *args: object) -> _T:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
