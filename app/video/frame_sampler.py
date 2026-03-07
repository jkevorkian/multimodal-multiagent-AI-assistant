from __future__ import annotations

import base64
import importlib.util
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


def _load_cv2():
    if importlib.util.find_spec("cv2") is None:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_sec: float
    cue: str
    image_uri: str | None = None


@dataclass(frozen=True)
class FrameSamplingResult:
    source: str
    estimated_duration_sec: float
    sample_fps: float
    requested_max_frames: int
    processed_frames: int
    budget_limited: bool
    frames: list[SampledFrame]
    extraction_mode: str = "decoded"


class VideoFrameSampler:
    """Decode-backed frame sampler with latency-budget guards.

    This sampler is strict by design:
    - OpenCV decode backend must be available.
    - A local path must be resolvable (remote URLs are downloaded to temp first).
    - At least one frame must be decoded, otherwise an explicit error is raised.
    """

    def __init__(
        self,
        latency_budget_ms: int = 1500,
        estimated_ms_per_frame: float = 12.0,
        max_window_sec: float = 120.0,
        max_frame_side: int = 1024,
        remote_fetch_timeout_sec: float = 20.0,
        max_remote_source_bytes: int = 120_000_000,
    ) -> None:
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be > 0")
        if estimated_ms_per_frame <= 0:
            raise ValueError("estimated_ms_per_frame must be > 0")
        if max_window_sec <= 0:
            raise ValueError("max_window_sec must be > 0")
        if max_frame_side <= 0:
            raise ValueError("max_frame_side must be > 0")
        if remote_fetch_timeout_sec <= 0:
            raise ValueError("remote_fetch_timeout_sec must be > 0")
        if max_remote_source_bytes <= 0:
            raise ValueError("max_remote_source_bytes must be > 0")
        self._latency_budget_ms = latency_budget_ms
        self._estimated_ms_per_frame = estimated_ms_per_frame
        self._max_window_sec = max_window_sec
        self._max_frame_side = max_frame_side
        self._remote_fetch_timeout_sec = remote_fetch_timeout_sec
        self._max_remote_source_bytes = max_remote_source_bytes

    def sample(self, video_uri: str, sample_fps: float = 1.0, max_frames: int = 32) -> FrameSamplingResult:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0")

        cv2 = _load_cv2()
        if cv2 is None:
            raise ValueError("opencv decode backend unavailable; install opencv-python for video processing")

        local_path, cleanup = self._materialize_local_video(video_uri)
        try:
            return self._sample_decoded_frames(
                cv2=cv2,
                video_uri=video_uri,
                local_path=local_path,
                sample_fps=sample_fps,
                max_frames=max_frames,
            )
        finally:
            if cleanup and cleanup.exists():
                try:
                    cleanup.unlink()
                except OSError:
                    pass

    def _sample_decoded_frames(
        self,
        *,
        cv2,
        video_uri: str,
        local_path: Path,
        sample_fps: float,
        max_frames: int,
    ) -> FrameSamplingResult:
        capture = cv2.VideoCapture(str(local_path))
        if not capture or not capture.isOpened():
            raise ValueError(f"unable to decode video source: {video_uri}")

        try:
            duration = self._resolve_duration_sec(
                capture=capture,
                cv2=cv2,
                video_uri=video_uri,
                sample_fps=sample_fps,
                max_frames=max_frames,
                local_path=local_path,
            )

            available_frames = max(1, int(duration * sample_fps))
            frame_budget = max(1, int(self._latency_budget_ms / self._estimated_ms_per_frame))
            requested_frames = min(max_frames, frame_budget, available_frames)
            step = 1.0 / sample_fps

            sampled_frames: list[SampledFrame] = []
            for index in range(requested_frames):
                timestamp = round(min(index * step, max(0.0, duration - 0.001)), 3)
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                sampled_frames.append(
                    SampledFrame(
                        index=index,
                        timestamp_sec=timestamp,
                        cue=f"frame_{index:03d}",
                        image_uri=self._to_frame_data_uri(cv2, frame),
                    )
                )

            if not sampled_frames:
                raise ValueError("decoded_frame_extraction_failed")

            processed_frames = len(sampled_frames)
            return FrameSamplingResult(
                source=video_uri,
                estimated_duration_sec=round(duration, 3),
                sample_fps=sample_fps,
                requested_max_frames=max_frames,
                processed_frames=processed_frames,
                budget_limited=(requested_frames < max_frames) or (processed_frames < max_frames),
                frames=sampled_frames,
                extraction_mode="decoded",
            )
        finally:
            capture.release()

    def _to_frame_data_uri(self, cv2, frame) -> str | None:
        payload = self._downscale_frame(cv2, frame)
        ok, encoded = cv2.imencode(".jpg", payload)
        if not ok:
            return None
        if hasattr(encoded, "tobytes"):
            encoded_bytes = encoded.tobytes()
        else:
            encoded_bytes = bytes(encoded)
        return f"data:image/jpeg;base64,{base64.b64encode(encoded_bytes).decode('ascii')}"

    def _downscale_frame(self, cv2, frame):
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return frame
        height, width = int(shape[0]), int(shape[1])
        max_side = max(height, width)
        if max_side <= self._max_frame_side:
            return frame
        scale = self._max_frame_side / max_side
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        interpolation = getattr(cv2, "INTER_AREA", 1)
        try:
            return cv2.resize(frame, target_size, interpolation=interpolation)
        except Exception:
            return frame

    def _resolve_duration_sec(
        self,
        *,
        capture,
        cv2,
        video_uri: str,
        sample_fps: float,
        max_frames: int,
        local_path: Path,
    ) -> float:
        parsed = urlparse(video_uri)
        query = parse_qs(parsed.query)
        for key in ("duration", "duration_sec", "seconds", "sec"):
            if key not in query:
                continue
            try:
                value = float(query[key][0])
            except (TypeError, ValueError, IndexError):
                continue
            return self._clamp_duration(value)

        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        measured_duration = (frame_count / native_fps) if native_fps > 0 and frame_count > 0 else 0.0
        if measured_duration > 0:
            return self._clamp_duration(measured_duration)

        try:
            size_bytes = max(1, local_path.stat().st_size)
        except OSError:
            size_bytes = 1
        # Conservative fallback duration when container metadata is incomplete.
        if size_bytes > 0:
            return self._clamp_duration(size_bytes / 400_000.0)

        return self._clamp_duration(max(8.0, max_frames / sample_fps))

    def _materialize_local_video(self, video_uri: str) -> tuple[Path, Path | None]:
        parsed = urlparse(video_uri)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix or ".mp4"
            payload = self._download_video_bytes(video_uri)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                temp_file.write(payload)
                temp_file.flush()
            finally:
                temp_file.close()
            return Path(temp_file.name), Path(temp_file.name)

        local_path = self._resolve_local_path(video_uri=video_uri, parsed=parsed)
        if local_path is None:
            raise ValueError(f"unsupported video URI scheme '{parsed.scheme}'")
        if not local_path.exists():
            raise ValueError(f"video source not found: {video_uri}")
        return local_path, None

    def _download_video_bytes(self, video_uri: str) -> bytes:
        request = Request(video_uri, headers={"User-Agent": "MMAA-VideoSampler/0.1"})
        with urlopen(request, timeout=self._remote_fetch_timeout_sec) as response:
            payload = response.read(self._max_remote_source_bytes + 1)
        if len(payload) > self._max_remote_source_bytes:
            raise ValueError("video source exceeds max allowed size for remote decode")
        if not payload:
            raise ValueError("empty video payload downloaded")
        return payload

    def _resolve_local_path(self, video_uri: str, parsed) -> Path | None:
        if parsed.scheme in {"http", "https", "s3"}:
            return None
        if parsed.scheme == "file":
            path_value = unquote(parsed.path)
            if len(path_value) >= 3 and path_value[0] == "/" and path_value[2] == ":":
                path_value = path_value[1:]
            return Path(path_value)
        candidate = Path(video_uri)
        return candidate

    def _clamp_duration(self, duration_sec: float) -> float:
        return max(3.0, min(self._max_window_sec, duration_sec))
