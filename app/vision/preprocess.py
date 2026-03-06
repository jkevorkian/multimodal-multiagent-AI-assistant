from __future__ import annotations

import base64
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

_ALLOWED_SCHEMES = {"", "file", "http", "https", "s3"}
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_REMOTE_FETCH_TIMEOUT_SEC = 10


@dataclass(frozen=True)
class ProcessedImage:
    image_uri: str
    inference_uri: str
    mime_type: str
    byte_size: int | None
    width: int | None
    height: int | None
    content_hash: str | None


class VisionPreprocessor:
    def __init__(self, max_source_bytes: int = 5_000_000) -> None:
        if max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be > 0")
        self._max_source_bytes = max_source_bytes

    async def preprocess(self, image_uri: str) -> ProcessedImage:
        if not image_uri.strip():
            raise ValueError("image_uri must not be empty")

        if image_uri.startswith("data:image/"):
            return self._process_data_uri(image_uri)

        parsed = urlparse(image_uri)
        scheme = parsed.scheme.lower()
        if self._is_windows_drive_path(image_uri):
            scheme = ""
        if scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"unsupported image URI scheme '{parsed.scheme}'")

        extension = Path(parsed.path or image_uri).suffix.lower()
        if extension and extension not in _ALLOWED_EXTENSIONS:
            raise ValueError(f"unsupported image extension '{extension}'")

        if scheme in {"http", "https"} and not extension:
            resolved = self._resolve_image_from_webpage(image_uri=image_uri)
            if resolved is None:
                raise ValueError(
                    "image_uri does not reference an image. Provide a direct image URL "
                    "or a webpage URL that contains a discoverable image."
                )
            data, mime_type = self._download_remote_image_bytes(resolved)
            if len(data) > self._max_source_bytes:
                raise ValueError("image exceeds max allowed size")
            width, height = self._extract_dimensions(data)
            return ProcessedImage(
                image_uri=image_uri,
                inference_uri=self._to_data_uri(data=data, mime_type=mime_type),
                mime_type=mime_type,
                byte_size=len(data),
                width=width,
                height=height,
                content_hash=hashlib.sha256(data).hexdigest(),
            )

        local_path = self._resolve_local_path(image_uri, scheme)
        if local_path is None:
            return ProcessedImage(
                image_uri=image_uri,
                inference_uri=image_uri,
                mime_type=self._mime_from_extension(extension),
                byte_size=None,
                width=None,
                height=None,
                content_hash=None,
            )

        data = local_path.read_bytes()
        if len(data) > self._max_source_bytes:
            raise ValueError("image exceeds max allowed size")
        width, height = self._extract_dimensions(data)
        return ProcessedImage(
            image_uri=image_uri,
            inference_uri=local_path.as_uri(),
            mime_type=self._mime_from_extension(extension),
            byte_size=len(data),
            width=width,
            height=height,
            content_hash=hashlib.sha256(data).hexdigest(),
        )

    def _process_data_uri(self, image_uri: str) -> ProcessedImage:
        try:
            header, encoded = image_uri.split(",", 1)
        except ValueError as exc:
            raise ValueError("invalid data URI format") from exc
        if ";base64" not in header:
            raise ValueError("data URI must be base64-encoded")
        mime_type = header.split(";", 1)[0].split(":", 1)[1]
        data = base64.b64decode(encoded, validate=True)
        if len(data) > self._max_source_bytes:
            raise ValueError("image exceeds max allowed size")
        width, height = self._extract_dimensions(data)
        return ProcessedImage(
            image_uri=image_uri,
            inference_uri=image_uri,
            mime_type=mime_type,
            byte_size=len(data),
            width=width,
            height=height,
            content_hash=hashlib.sha256(data).hexdigest(),
        )

    def _resolve_local_path(self, image_uri: str, scheme: str) -> Path | None:
        if scheme in {"http", "https", "s3"}:
            return None
        if scheme == "file":
            parsed = urlparse(image_uri)
            path_value = unquote(parsed.path)
            if len(path_value) >= 3 and path_value[0] == "/" and path_value[2] == ":":
                path_value = path_value[1:]
            return Path(path_value)
        return Path(image_uri)

    def _is_windows_drive_path(self, image_uri: str) -> bool:
        return bool(re.match(r"^[A-Za-z]:[\\/]", image_uri))

    def _mime_from_extension(self, extension: str) -> str:
        if extension in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if extension == ".png":
            return "image/png"
        if extension == ".webp":
            return "image/webp"
        if extension == ".gif":
            return "image/gif"
        if extension == ".bmp":
            return "image/bmp"
        if extension in {".tif", ".tiff"}:
            return "image/tiff"
        return "image/*"

    def _extract_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
            width = struct.unpack(">I", data[16:20])[0]
            height = struct.unpack(">I", data[20:24])[0]
            return width, height
        return None, None

    def _resolve_image_from_webpage(self, image_uri: str) -> str | None:
        payload, content_type = self._fetch_url_bytes(image_uri)
        content_type_main = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        if content_type_main.startswith("image/"):
            return image_uri

        html_text = payload.decode("utf-8", errors="ignore")
        meta_candidate = self._extract_meta_image_candidate(html_text)
        if meta_candidate:
            return urljoin(image_uri, meta_candidate)

        pattern = r"https?://[^\"'\s>]+\.(?:png|jpg|jpeg|webp|gif|bmp|tif|tiff)"
        matches = re.findall(pattern, html_text, flags=re.IGNORECASE)
        if matches:
            return matches[0]

        relative_pattern = r"(?i)(?:src|href)\s*=\s*[\"']([^\"']+\.(?:png|jpg|jpeg|webp|gif|bmp|tif|tiff))(?:\?[^\"']*)?[\"']"
        relative_matches = re.findall(relative_pattern, html_text)
        if relative_matches:
            return urljoin(image_uri, relative_matches[0])
        return None

    def _extract_meta_image_candidate(self, html_text: str) -> str | None:
        patterns = [
            r"(?is)<meta[^>]+(?:property|name)\s*=\s*[\"'](?:og:image|twitter:image|twitter:image:src)[\"'][^>]+content\s*=\s*[\"']([^\"']+)[\"']",
            r"(?is)<meta[^>]+content\s*=\s*[\"']([^\"']+)[\"'][^>]+(?:property|name)\s*=\s*[\"'](?:og:image|twitter:image|twitter:image:src)[\"']",
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
        return None

    def _download_remote_image_bytes(self, image_uri: str) -> tuple[bytes, str]:
        payload, content_type = self._fetch_url_bytes(image_uri)
        content_type_main = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
        if not content_type_main.startswith("image/"):
            extension = Path(urlparse(image_uri).path).suffix.lower()
            mime_type = self._mime_from_extension(extension)
            if mime_type == "image/*":
                raise ValueError("resolved resource is not an image")
            content_type_main = mime_type
        return payload, content_type_main

    def _fetch_url_bytes(self, image_uri: str) -> tuple[bytes, str]:
        request = Request(image_uri, headers={"User-Agent": "MMAA-Vision/0.1"})
        with urlopen(request, timeout=_REMOTE_FETCH_TIMEOUT_SEC) as response:
            payload = response.read(self._max_source_bytes + 1)
            content_type = response.headers.get("Content-Type", "")
        return payload, content_type

    def _to_data_uri(self, data: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
