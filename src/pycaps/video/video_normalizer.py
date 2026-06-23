from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

from pycaps.logger import logger


FRAME_RATE_RELATIVE_TOLERANCE = 0.001


@dataclass(frozen=True)
class VideoFrameRate:
    text: str
    value: float


@dataclass(frozen=True)
class VideoMetadata:
    r_frame_rate: VideoFrameRate | None
    avg_frame_rate: VideoFrameRate | None
    duration: float | None
    nb_frames: int | None


@dataclass(frozen=True)
class NormalizedVideoInput:
    path: str
    is_temporary: bool


def parse_frame_rate(value: str | None) -> VideoFrameRate | None:
    if value is None:
        return None

    text = value.strip()
    if not text or text == "N/A" or text == "0/0":
        return None

    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return None
        if denominator == 0:
            return None
        frame_rate = numerator / denominator
    else:
        try:
            frame_rate = float(text)
        except ValueError:
            return None

    if frame_rate <= 0:
        return None
    return VideoFrameRate(text=text, value=frame_rate)


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def probe_video_metadata(video_path: str | os.PathLike[str]) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "ffprobe failed to inspect video metadata.")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON.") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("ffprobe did not return a video stream.")

    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError("ffprobe returned an invalid video stream payload.")

    return VideoMetadata(
        r_frame_rate=parse_frame_rate(str(stream.get("r_frame_rate") or "")),
        avg_frame_rate=parse_frame_rate(str(stream.get("avg_frame_rate") or "")),
        duration=_parse_optional_float(stream.get("duration")),
        nb_frames=_parse_optional_int(stream.get("nb_frames")),
    )


def should_normalize_to_cfr(metadata: VideoMetadata) -> bool:
    if metadata.r_frame_rate is None or metadata.avg_frame_rate is None:
        return False

    reference = metadata.r_frame_rate.value
    if reference <= 0:
        return False

    relative_delta = abs(reference - metadata.avg_frame_rate.value) / reference
    return relative_delta > FRAME_RATE_RELATIVE_TOLERANCE


def _run_ffmpeg_normalization(
    *,
    input_path: str,
    output_path: str,
    target_frame_rate: str,
    copy_audio: bool,
) -> subprocess.CompletedProcess[str]:
    audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "192k"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"fps={target_frame_rate}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        *audio_args,
        "-movflags",
        "+faststart",
        output_path,
    ]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def normalize_video_to_cfr_if_needed(
    input_video_path: str | os.PathLike[str],
) -> NormalizedVideoInput:
    input_path = str(input_video_path)
    metadata = probe_video_metadata(input_path)

    if not should_normalize_to_cfr(metadata):
        return NormalizedVideoInput(path=input_path, is_temporary=False)
    if metadata.r_frame_rate is None:
        return NormalizedVideoInput(path=input_path, is_temporary=False)

    suffix = Path(input_path).suffix or ".mp4"
    fd, output_path = tempfile.mkstemp(prefix="pycaps_cfr_", suffix=suffix)
    os.close(fd)

    try:
        result = _run_ffmpeg_normalization(
            input_path=input_path,
            output_path=output_path,
            target_frame_rate=metadata.r_frame_rate.text,
            copy_audio=True,
        )
        if result.returncode != 0:
            logger().warning(
                "CFR normalization with audio copy failed, retrying with AAC audio: %s",
                (result.stderr or "").strip(),
            )
            result = _run_ffmpeg_normalization(
                input_path=input_path,
                output_path=output_path,
                target_frame_rate=metadata.r_frame_rate.text,
                copy_audio=False,
            )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "ffmpeg failed to normalize video to CFR.")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("ffmpeg produced an empty normalized video.")

        logger().debug(
            "Normalized VFR input to CFR: input=%s output=%s fps=%s",
            input_path,
            output_path,
            metadata.r_frame_rate.text,
        )
        return NormalizedVideoInput(path=output_path, is_temporary=True)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
