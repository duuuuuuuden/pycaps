from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

from pycaps.video.video_generator import VideoGenerator
from pycaps.video.video_normalizer import (
    NormalizedVideoInput,
    VideoFrameRate,
    VideoMetadata,
    normalize_video_to_cfr_if_needed,
    parse_frame_rate,
)


class VideoNormalizerTests(unittest.TestCase):
    def test_parse_frame_rate_values(self):
        self.assertAlmostEqual(parse_frame_rate("24/1").value, 24.0)
        self.assertAlmostEqual(parse_frame_rate("24000/1001").value, 23.976023976)
        self.assertIsNone(parse_frame_rate("0/0"))
        self.assertIsNone(parse_frame_rate("N/A"))

    def test_cfr_input_returns_original_path(self):
        metadata = VideoMetadata(
            r_frame_rate=VideoFrameRate(text="24/1", value=24.0),
            avg_frame_rate=VideoFrameRate(text="24/1", value=24.0),
            duration=1.0,
            nb_frames=24,
        )

        with (
            patch("pycaps.video.video_normalizer.probe_video_metadata", return_value=metadata),
            patch("pycaps.video.video_normalizer._run_ffmpeg_normalization") as run_ffmpeg,
        ):
            result = normalize_video_to_cfr_if_needed("input.mp4")

        self.assertEqual(result.path, "input.mp4")
        self.assertFalse(result.is_temporary)
        run_ffmpeg.assert_not_called()

    def test_vfr_input_normalizes_with_nominal_frame_rate(self):
        metadata = VideoMetadata(
            r_frame_rate=VideoFrameRate(text="24/1", value=24.0),
            avg_frame_rate=VideoFrameRate(text="12156/589", value=20.6383701188455),
            duration=49.083333,
            nb_frames=1013,
        )
        calls = []

        def run_ffmpeg(**kwargs):
            calls.append(kwargs)
            with open(kwargs["output_path"], "wb") as output_file:
                output_file.write(b"normalized-video")
            return SimpleNamespace(returncode=0, stderr="")

        with (
            patch("pycaps.video.video_normalizer.probe_video_metadata", return_value=metadata),
            patch("pycaps.video.video_normalizer._run_ffmpeg_normalization", side_effect=run_ffmpeg),
        ):
            result = normalize_video_to_cfr_if_needed("input.mp4")

        self.addCleanup(lambda: os.path.exists(result.path) and os.remove(result.path))
        self.assertTrue(result.is_temporary)
        self.assertTrue(os.path.exists(result.path))
        self.assertEqual(calls[0]["target_frame_rate"], "24/1")
        self.assertTrue(calls[0]["copy_audio"])

    def test_vfr_input_retries_with_aac_when_audio_copy_fails(self):
        metadata = VideoMetadata(
            r_frame_rate=VideoFrameRate(text="24/1", value=24.0),
            avg_frame_rate=VideoFrameRate(text="20/1", value=20.0),
            duration=1.0,
            nb_frames=20,
        )
        copy_audio_values = []

        def run_ffmpeg(**kwargs):
            copy_audio_values.append(kwargs["copy_audio"])
            if kwargs["copy_audio"]:
                return SimpleNamespace(returncode=1, stderr="copy failed")
            with open(kwargs["output_path"], "wb") as output_file:
                output_file.write(b"normalized-video")
            return SimpleNamespace(returncode=0, stderr="")

        with (
            patch("pycaps.video.video_normalizer.probe_video_metadata", return_value=metadata),
            patch("pycaps.video.video_normalizer._run_ffmpeg_normalization", side_effect=run_ffmpeg),
        ):
            result = normalize_video_to_cfr_if_needed("input.mp4")

        self.addCleanup(lambda: os.path.exists(result.path) and os.remove(result.path))
        self.assertEqual(copy_audio_values, [True, False])
        self.assertTrue(result.is_temporary)

    def test_close_removes_temporary_normalized_input_video(self):
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        generator = VideoGenerator()
        generator._normalized_input_video_path = temp_path
        generator.close()

        self.assertFalse(os.path.exists(temp_path))

    def test_start_removes_temporary_normalized_video_on_failure(self):
        fd, source_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        fd, normalized_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(source_path) and os.remove(source_path))

        generator = VideoGenerator()
        with (
            patch(
                "pycaps.video.video_generator.normalize_video_to_cfr_if_needed",
                return_value=NormalizedVideoInput(
                    path=normalized_path,
                    is_temporary=True,
                ),
            ),
            patch("movielite.VideoClip", side_effect=RuntimeError("load failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                generator.start(source_path, "output.mp4")

        self.assertFalse(os.path.exists(normalized_path))


if __name__ == "__main__":
    unittest.main()
