#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reading Video — create a text-synchronized video book from audio/video
and a plain-text transcript.

Features
--------
- Extract audio from a video (or accept an audio file).
- Align text with audio using stable-whisper (stable-ts).
- Paginate text to fit an adjustable "page" size.
- Render video frames with word highlighting in sync with audio.
- Minimal Tkinter UI with progress logging.

Requirements
------------
- Python 3.9+
- Packages: stable-ts (stable-whisper), opencv-python, pillow, moviepy, numpy

Install
-------
pip install \
  git+https://github.com/jianfch/stable-ts.git \
  opencv-python pillow moviepy numpy imageio-ffmpeg

macOS notes
-----------
- SDL runs in headless (dummy) mode to avoid Tkinter conflicts.
- H.264 ('avc1') is used on macOS for better compatibility.
- If MoviePy has ffmpeg issues, install `imageio-ffmpeg`.
- Alternative launch (if GUI issues persist):
  pythonw videobook_ui.py

License
-------
MIT
"""

from __future__ import annotations

import logging
import os
import pickle
import platform
import queue
import shutil
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# Environment preparation for macOS (SDL/Tkinter conflicts)
if platform.system() == "Darwin":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

# ffmpeg discovery and environment setup
FFMPEG_PATH = shutil.which("ffmpeg")
if not FFMPEG_PATH:
    for _p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.exists(_p):
            FFMPEG_PATH = _p
            break
    if not FFMPEG_PATH:
        try:
            import imageio_ffmpeg  # type: ignore

            FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG_PATH = None

if FFMPEG_PATH:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
    _dir = os.path.dirname(FFMPEG_PATH)
    if _dir and _dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_dir}:{os.environ.get('PATH', '')}"

# Hide pygame prompt if imported by dependencies
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# Tkinter must import before heavy multimedia libs to reduce conflicts
from tkinter import (  # noqa: E402
    BOTH,
    DISABLED,
    END,
    LEFT,
    NORMAL,
    N,
    E,
    S,
    W,
    WORD,
    StringVar,
    IntVar,
    Tk,
)
from tkinter import filedialog, messagebox, ttk  # noqa: E402
from tkinter.scrolledtext import ScrolledText  # noqa: E402

# Third-party imports
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

try:
    import stable_whisper  # noqa: E402
except Exception as _err:
    # Defer user notification to UI where we can show a dialog
    stable_whisper = None  # type: ignore

try:
    from moviepy.editor import AudioFileClip, VideoFileClip  # noqa: E402
    try:
        from moviepy.config import change_settings  # noqa: E402

        if FFMPEG_PATH:
            change_settings({"FFMPEG_BINARY": FFMPEG_PATH})
    except Exception:
        pass
except Exception as _err:
    AudioFileClip = None  # type: ignore
    VideoFileClip = None  # type: ignore

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("videobook")
_HANDLER = logging.StreamHandler(sys.stdout)
_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
)
_HANDLER.setFormatter(_FORMAT)
LOGGER.addHandler(_HANDLER)
LOGGER.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class WordStamp:
    """A single word with timing information."""

    word: str
    start: float
    end: float


# -----------------------------------------------------------------------------
# Core creator
# -----------------------------------------------------------------------------

class VideoBookCreator:
    """
    Create a videobook from an audio/video file and a text file.

    The creator:
    - extracts audio (or accepts audio directly),
    - aligns text to get per-word timestamps,
    - paginates words to fit a configurable page,
    - renders frames with a highlight for the active word,
    - muxes video and audio into the final file.

    Parameters
    ----------
    video_file:
        Path to the source media (audio or video).
    text_file:
        Path to the plain text file to align.
    output_video:
        Target path for the final video.
    log_callback:
        Callable to receive log messages for the UI.
    progress_callback:
        Callable to receive integer progress (0..100).
    width, height, fps:
        Output frame size and FPS.
    sync_offset:
        Global sync shift in seconds (negative means earlier highlight).
    font_path:
        Optional path to a TTF/OTF font file to use for text rendering.
    margin, font_size, line_spacing, page_num_font_size:
        Layout parameters (auto-scaled if not provided).
    text_color, page_color, highlight_color:
        Hex color strings for text, background, and highlight.
    language:
        Language code for alignment (e.g., "en", "ru").
    """

    def __init__(
        self,
        video_file: str,
        text_file: str,
        output_video: str,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs: object,
    ) -> None:
        # Files
        self.video_file = video_file
        self.text_file = text_file
        self.output_video = output_video
        self.temp_audio_file = "temp_extracted_audio.wav"
        self.temp_video_file = "temp_silent_video.mp4"

        # UI callbacks
        self.log_callback = log_callback or (lambda m: LOGGER.info("%s", m))
        self.progress_callback = progress_callback

        # Control flag
        self.stop_processing = False

        # Model
        self.model_cache_file = "whisper_model_cache.pkl"
        self.model = None

        # Video params
        self.width = int(kwargs.get("width", 1160))  # type: ignore[arg-type]
        self.height = int(kwargs.get("height", 1534))  # type: ignore[arg-type]
        self.fps = int(kwargs.get("fps", 30))  # type: ignore[arg-type]
        self.sync_offset = float(kwargs.get("sync_offset", 0.0))  # type: ignore

        # Layout scaling
        base_width = 1160
        scale = self.width / base_width

        self.margin = int(kwargs.get("margin", int(80 * scale)))  # type: ignore
        self.font_path = kwargs.get("font_path", None)
        self.font_size = int(kwargs.get("font_size", int(80 * scale)))  # type: ignore
        self.line_spacing = int(
            kwargs.get("line_spacing", int(30 * scale))  # type: ignore
        )
        self.page_num_font_size = int(
            kwargs.get("page_num_font_size", int(50 * scale))  # type: ignore
        )

        self.text_color = str(kwargs.get("text_color", "#363636"))
        self.page_color = str(kwargs.get("page_color", "#F5F5DC"))
        self.highlight_color = str(kwargs.get("highlight_color", "#FFD700"))
        self.language = str(kwargs.get("language", "ru"))

        self.font: ImageFont.FreeTypeFont
        self.page_num_font: ImageFont.FreeTypeFont
        self._load_fonts()

    # --------------------------- helpers / logging ---------------------------

    def _log(self, msg: str) -> None:
        """Send a log message to the UI and to the logger."""
        if self.log_callback:
            self.log_callback(msg)
        else:
            LOGGER.info("%s", msg)

    def _progress(self, val: int) -> None:
        """Report progress to the UI if provided."""
        if self.progress_callback:
            self.progress_callback(int(val))

    # --------------------------- initialization -----------------------------

    def _load_fonts(self) -> None:
        """Load fonts, falling back to system or default fonts."""
        try:
            fp = None if self.font_path in (None, "") else str(self.font_path)
            if fp and os.path.exists(fp):
                self.font = ImageFont.truetype(fp, self.font_size)
                self.page_num_font = ImageFont.truetype(
                    fp, self.page_num_font_size
                )
                return
            raise OSError("Font path not provided or not found.")
        except Exception:
            self._log(
                "⚠️  Font not found; falling back to system/default fonts."
            )
            for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
                try:
                    self.font = ImageFont.truetype(name, self.font_size)
                    self.page_num_font = ImageFont.truetype(
                        name, self.page_num_font_size
                    )
                    return
                except Exception:
                    continue
            # As a last resort
            self.font = ImageFont.load_default()
            self.page_num_font = ImageFont.load_default()

    # ------------------------------- pipeline -------------------------------

    def run(self) -> None:
        """Run the complete creation pipeline."""
        start_time = datetime.now()
        self._log("🚀 [START] Building videobook")
        self._log(f"📐 Page size: {self.width}x{self.height}")
        self._log(f"📁 Output: {os.path.abspath(self.output_video)}")

        for tmp in (self.temp_audio_file, self.temp_video_file, "temp-audio.m4a"):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                    self._log(f"   Removed leftover temp file: {tmp}")
                except OSError:
                    pass

        try:
            self._extract_audio()
            words = self._align_text_and_audio()
            pages = self._split_text_into_pages(words)
            self._create_video_frames(pages)
            self._add_audio_to_video()
        except Exception as exc:
            self._log(f"❌ [ERROR] Fatal pipeline error: {exc}")
            raise
        finally:
            self._cleanup()
            end_time = datetime.now()
            self._log(f"\n🎉 [DONE] Elapsed: {end_time - start_time}")
            if os.path.exists(self.output_video):
                size_mb = os.path.getsize(self.output_video) / (1024 * 1024)
                self._log(
                    f"📁 Saved: {os.path.abspath(self.output_video)} "
                    f"({size_mb:.1f} MB)"
                )

    # ------------------------------- stages ---------------------------------

    def _extract_audio(self) -> None:
        """Extract or normalize audio to 16kHz PCM WAV."""
        self._log("\n[1/5] 🔉 Preparing audio...")
        self._progress(10)

        if not os.path.exists(self.video_file):
            raise FileNotFoundError(f"Input not found: {self.video_file}")

        if platform.system() == "Darwin":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        try:
            if FFMPEG_PATH:
                try:
                    from moviepy.config import change_settings  # type: ignore

                    change_settings({"FFMPEG_BINARY": FFMPEG_PATH})
                    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
                    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
                    _d = os.path.dirname(FFMPEG_PATH)
                    if _d and _d not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = f"{_d}:{os.environ.get('PATH', '')}"
                    self._log(f"   ffmpeg: {FFMPEG_PATH}")
                except Exception as e_set:
                    self._log(f"   ⚠️ ffmpeg setup skipped: {e_set}")
        except Exception as _e:
            self._log(f"   ⚠️ Could not refresh ffmpeg settings: {_e}")

        ext = os.path.splitext(self.video_file)[1].lower()
        audio_exts = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"}

        if AudioFileClip is None or VideoFileClip is None:
            raise RuntimeError(
                "MoviePy is unavailable. Please reinstall moviepy and "
                "imageio-ffmpeg."
            )

        if ext in audio_exts:
            with AudioFileClip(self.video_file) as audio:
                audio.write_audiofile(
                    self.temp_audio_file,
                    codec="pcm_s16le",
                    fps=16000,
                    logger=None,
                )
            self._log("✅ Audio normalized to 16kHz WAV.")
        else:
            with VideoFileClip(self.video_file) as video:
                if video.audio is None:
                    raise ValueError("No audio track found in input video.")
                video.audio.write_audiofile(
                    self.temp_audio_file,
                    codec="pcm_s16le",
                    fps=16000,
                    logger=None,
                )
            self._log("✅ Audio extracted from video.")

    def _load_or_cache_model(self) -> None:
        """Load stable-whisper model from cache or download it."""
        if stable_whisper is None:
            raise RuntimeError(
                "stable-whisper (stable-ts) is not installed or failed "
                "to import."
            )

        if os.path.exists(self.model_cache_file):
            try:
                self._log("📦 Loading model from cache...")
                with open(self.model_cache_file, "rb") as f:
                    self.model = pickle.load(f)
                self._log("✅ Model loaded from cache.")
                return
            except Exception as exc:
                self._log(f"⚠️ Cache load failed: {exc}")

        self._log("🌐 Downloading Whisper model (first run may take a while)...")
        self.model = stable_whisper.load_model("base")
        try:
            with open(self.model_cache_file, "wb") as f:
                pickle.dump(self.model, f)
            self._log("💾 Model cached for next runs.")
        except Exception as exc:
            self._log(f"⚠️ Could not cache model: {exc}")

    def _align_text_and_audio(self) -> List[WordStamp]:
        """Align audio and text to get per-word timestamps."""
        self._log("\n[2/5] 🧠 Aligning text and audio...")
        self._progress(20)

        if not os.path.exists(self.text_file):
            raise FileNotFoundError(f"Text file not found: {self.text_file}")

        if self.model is None:
            self._load_or_cache_model()

        with open(self.text_file, "r", encoding="utf-8") as f:
            text_content = f.read()

        assert self.model is not None
        result = self.model.align(
            self.temp_audio_file,
            text_content,
            language=self.language,
        )

        words: List[WordStamp] = []
        for seg in result.segments:
            for w in seg.words:
                words.append(WordStamp(word=w.word, start=w.start, end=w.end))

        if not words:
            raise ValueError(
                "Alignment produced no words. Text may not match the audio."
            )

        self._log(f"✅ Alignment done. Words: {len(words)}")
        self._progress(40)
        return words

    def _split_text_into_pages(
        self, words: List[WordStamp]
    ) -> List[List[WordStamp]]:
        """
        Split words into pages so lines fit the configured page width/height.
        """
        self._log("\n[3/5] 📖 Paginating...")
        self._progress(50)

        pages: List[List[WordStamp]] = []
        current_page: List[WordStamp] = []

        tmp_img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(tmp_img)

        max_w = self.width - 2 * self.margin
        max_h = (
            self.height
            - 2 * self.margin
            - self.page_num_font_size
            - self.line_spacing * 2
        )

        cur_y = 0
        cur_line = ""
        cur_line_words: List[WordStamp] = []

        for wi in words:
            test_line = f"{cur_line} {wi.word}".strip()
            if draw.textlength(test_line, font=self.font) <= max_w:
                cur_line = test_line
                cur_line_words.append(wi)
            else:
                if cur_line:
                    next_y = cur_y + self.font_size + self.line_spacing
                    if next_y > max_h:
                        if current_page:
                            pages.append(current_page)
                        current_page = []
                        cur_y = 0
                    else:
                        cur_y = next_y
                    current_page.extend(cur_line_words)
                cur_line = wi.word
                cur_line_words = [wi]

        if cur_line_words:
            current_page.extend(cur_line_words)
        if current_page:
            pages.append(current_page)

        self._log(f"✅ Pages: {len(pages)}")
        return pages

    def _map_time_to_content(
        self, pages: List[List[WordStamp]], total_frames: int
    ) -> List[Tuple[int, Optional[WordStamp]]]:
        """
        Precompute (page_index, active_word) for every frame time.
        """
        page_map: List[int] = []
        word_map: List[Optional[WordStamp]] = []

        page_idx = 0
        for i in range(total_frames):
            current_time = max(0.0, (i / self.fps) + self.sync_offset)

            if page_idx < len(pages) - 1:
                next_start = pages[page_idx + 1][0].start
                if current_time >= next_start:
                    page_idx += 1
            page_map.append(page_idx)

            active: Optional[WordStamp] = None
            for wi in pages[page_idx]:
                if wi.start <= current_time:
                    active = wi
                else:
                    break
            word_map.append(active)

        return list(zip(page_map, word_map))

    def _render_page_image(
        self,
        page_words: List[WordStamp],
        highlighted: Optional[WordStamp],
        page_num: int,
        total_pages: int,
    ) -> Image.Image:
        """
        Render a single page image with an optional highlighted word.
        """
        img = Image.new("RGB", (self.width, self.height), color=self.page_color)
        draw = ImageDraw.Draw(img)

        y = self.margin
        max_w = self.width - 2 * self.margin

        idx = 0
        while idx < len(page_words):
            line_words: List[WordStamp] = []
            line_text = ""

            while idx < len(page_words):
                wi = page_words[idx]
                candidate = f"{line_text} {wi.word}".strip()
                if draw.textlength(candidate, font=self.font) <= max_w:
                    line_text = candidate
                    line_words.append(wi)
                    idx += 1
                else:
                    break

            if not line_words:
                line_words = [page_words[idx]]
                idx += 1

            # Detect naive headings (simple heuristics)
            is_heading = False
            if line_words:
                first = line_words[0].word.strip()
                upper = line_text.upper()
                if (first.replace(".", "").isdigit() and "." in first) or (
                    "CHAPTER" in upper or "ГЛАВА" in upper
                ):
                    is_heading = True

            x = self.margin
            for wi in line_words:
                word = wi.word
                cur_font = self.font

                if is_heading:
                    try:
                        size = min(self.font_size + 10, int(self.font_size * 1.2))
                        if self.font_path and os.path.exists(str(self.font_path)):
                            cur_font = ImageFont.truetype(
                                str(self.font_path), size
                            )
                        else:
                            cur_font = self.font
                    except Exception:
                        cur_font = self.font

                bbox = draw.textbbox((x, y), word, font=cur_font)

                if is_heading:
                    draw.rectangle(
                        [(bbox[0] - 5, bbox[1] - 3), (bbox[2] + 5, bbox[3] + 3)],
                        fill="#E8E8E8",
                    )
                    if highlighted and wi.start == highlighted.start:
                        draw.rectangle(
                            [
                                (bbox[0] - 3, bbox[1] - 2),
                                (bbox[2] + 3, bbox[3] + 2),
                            ],
                            fill=self.highlight_color,
                        )
                    draw.text((x, y), word, font=cur_font, fill="#2C3E50")
                else:
                    if highlighted and wi.start == highlighted.start:
                        draw.rectangle(
                            [
                                (bbox[0] - 3, bbox[1] - 2),
                                (bbox[2] + 3, bbox[3] + 2),
                            ],
                            fill=self.highlight_color,
                        )
                    draw.text((x, y), word, font=cur_font, fill=self.text_color)

                x += draw.textlength(f"{word} ", font=self.font)

            y += self.font_size + self.line_spacing
            if is_heading:
                y += int(self.line_spacing * 0.5)

        # Page number
        pager = f"{page_num} / {total_pages}"
        pager_w = draw.textlength(pager, font=self.page_num_font)
        draw.text(
            (
                self.width - self.margin - pager_w,
                self.height - self.margin - self.page_num_font_size,
            ),
            pager,
            font=self.page_num_font,
            fill=self.text_color,
        )
        return img

    def _create_video_frames(self, pages: List[List[WordStamp]]) -> None:
        """Render all frames and write a silent video file."""
        self._log("\n[4/5] 🎬 Rendering frames...")
        self._progress(60)

        last_end = pages[-1][-1].end if pages and pages[-1] else 0.0
        total_frames = int(round(last_end * self.fps))
        if total_frames <= 0:
            raise ValueError("Zero duration. Cannot build video.")

        self._log(f"📊 Frames: {total_frames}")
        self._log(f"⏱ Duration: {last_end:.1f}s")
        self._log(f"🖼 Size: {self.width}x{self.height}")

        if platform.system() == "Darwin":
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            self.temp_video_file, fourcc, self.fps, (self.width, self.height)
        )

        if not writer.isOpened():
            self._log("⚠️ Fallback to MJPG/AVI...")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.temp_video_file = self.temp_video_file.replace(".mp4", ".avi")
            writer = cv2.VideoWriter(
                self.temp_video_file, fourcc, self.fps, (self.width, self.height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not initialize VideoWriter.")

        frame_cache: Dict[Tuple[int, float], np.ndarray] = {}
        time_map = self._map_time_to_content(pages, total_frames)

        report_every = max(1, total_frames // 20)
        last_report = datetime.now()

        try:
            for i in range(total_frames):
                if self.stop_processing:
                    self._log("⏹ Stopped by user.")
                    break

                if i % report_every == 0 or i == 0:
                    now = datetime.now()
                    if i > 0:
                        elapsed = (now - last_report).total_seconds()
                        fps_act = report_every / elapsed if elapsed > 0 else 0.0
                        eta = (total_frames - i) / fps_act if fps_act > 0 else 0.0
                        eta_m, eta_s = int(eta // 60), int(eta % 60)
                        pct = int(100 * i / total_frames)
                        self._log(
                            "   Progress: %3d%% (%d/%d) | Speed: %.1f fps | "
                            "ETA: %d:%02d",
                            # logging supports %-style, but we format here:
                        )
                        self._log(
                            f"   Progress: {pct:3d}% ({i}/{total_frames}) | "
                            f"Speed: {fps_act:.1f} fps | ETA: {eta_m}:{eta_s:02d}"
                        )
                    else:
                        self._log(f"   Rendering {total_frames} frames...")
                    last_report = now
                    self._progress(60 + int(20 * (i / total_frames)))

                page_idx, active = time_map[i]
                highlight_t = active.start if active else -1.0
                key = (page_idx, highlight_t)

                if key in frame_cache:
                    frame = frame_cache[key]
                else:
                    img = self._render_page_image(
                        pages[page_idx], active, page_idx + 1, len(pages)
                    )
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    frame_cache[key] = frame

                writer.write(frame)
        finally:
            writer.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if not self.stop_processing:
            self._log(
                "\n✅ Silent video created. Unique rendered frames: "
                f"{len(frame_cache)}"
            )
        self._progress(80)

    def _add_audio_to_video(self) -> None:
        """Mux the audio track into the rendered silent video."""
        self._log("\n[5/5] 🎼 Muxing audio...")
        self._progress(85)

        if not os.path.exists(self.temp_video_file):
            self._log("❌ Missing temp video. Cannot mux.")
            return

        if platform.system() == "Darwin":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        if AudioFileClip is None or VideoFileClip is None:
            raise RuntimeError("MoviePy is unavailable.")

        silent = VideoFileClip(self.temp_video_file)
        audio = AudioFileClip(self.temp_audio_file)

        try:
            try:
                final = silent.with_audio(audio)  # MoviePy 2.x
            except AttributeError:
                final = silent.set_audio(audio)  # MoviePy 1.x

            try:
                from moviepy.config import change_settings  # type: ignore

                if FFMPEG_PATH:
                    change_settings({"FFMPEG_BINARY": FFMPEG_PATH})
                    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
                    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
            except Exception:
                pass

            final.write_videofile(
                self.output_video,
                codec="libx264",
                audio_codec="aac",
                fps=self.fps,
                audio_fps=44100,
                logger=None,
                temp_audiofile="temp-audio.m4a",
                threads=4,
            )
        finally:
            try:
                silent.close()
            except Exception:
                pass
            try:
                audio.close()
            except Exception:
                pass
            try:
                final.close()  # type: ignore
            except Exception:
                pass

        self._log("✅ Final video written.")
        self._progress(100)

    def _cleanup(self) -> None:
        """Remove temporary files created during processing."""
        self._log("\n🧹 [CLEANUP] Removing temp files...")
        for fpath in (
            self.temp_audio_file,
            self.temp_video_file,
            "temp-audio.m4a",
            "temp-audio.m4v",
        ):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    self._log(f"   - Removed '{fpath}'.")
                except OSError as exc:
                    self._log(f"   - Could not remove '{fpath}': {exc}")


# -----------------------------------------------------------------------------
# Tk UI
# -----------------------------------------------------------------------------

class VideoBookApp:
    """Minimal Tkinter UI for creating videobooks."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Reading Video Maker")
        self.root.geometry("900x700")

        if platform.system() == "Darwin":
            self.root.minsize(800, 600)

        self.video_file = StringVar()
        self.text_file = StringVar()
        self.output_file = StringVar(value="final_video_book.mp4")
        self.font_file = StringVar()

        self.width = IntVar(value=1160)
        self.height = IntVar(value=1534)
        self.fps = IntVar(value=30)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.current_creator: Optional[VideoBookCreator] = None
        self.processing = False

        self._create_widgets()

        # System info
        self._log_gui(f"💻 OS: {platform.system()} {platform.release()}")
        self._log_gui(f"🐍 Python: {sys.version.split()[0]}")
        if platform.system() == "Darwin":
            self._log_gui("🍎 macOS detected — SDL in headless mode.")
        if FFMPEG_PATH:
            self._log_gui(f"🎬 ffmpeg: {FFMPEG_PATH}")
        else:
            self._log_gui("⚠️ ffmpeg not found. Please install it.")

        self._update_logs()

        # Import failure notifications (late UI dialogs)
        if stable_whisper is None:
            messagebox.showerror(
                "Import error",
                "stable-whisper (stable-ts) is not installed or failed "
                "to import.\nInstall it with:\n"
                "pip install "
                "git+https://github.com/jianfch/stable-ts.git",
            )
        if AudioFileClip is None or VideoFileClip is None:
            messagebox.showerror(
                "Import error",
                "MoviePy is not available. Reinstall moviepy and "
                "imageio-ffmpeg.",
            )

    # ------------------------------- UI build --------------------------------

    def _create_widgets(self) -> None:
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky=(W, E, N, S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        files = ttk.LabelFrame(main, text="📁 Files", padding="10")
        files.grid(row=0, column=0, columnspan=2, sticky=(W, E), pady=5)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Audio/Video:").grid(
            row=0, column=0, sticky=W, padx=5, pady=5
        )
        ttk.Entry(files, textvariable=self.video_file, width=50).grid(
            row=0, column=1, sticky=(W, E), padx=5, pady=5
        )
        ttk.Button(
            files, text="Choose", command=self._select_video_file
        ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(files, text="Book text:").grid(
            row=1, column=0, sticky=W, padx=5, pady=5
        )
        ttk.Entry(files, textvariable=self.text_file, width=50).grid(
            row=1, column=1, sticky=(W, E), padx=5, pady=5
        )
        ttk.Button(
            files, text="Choose", command=self._select_text_file
        ).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(files, text="Save as:").grid(
            row=2, column=0, sticky=W, padx=5, pady=5
        )
        ttk.Entry(files, textvariable=self.output_file, width=50).grid(
            row=2, column=1, sticky=(W, E), padx=5, pady=5
        )
        ttk.Button(
            files, text="Choose", command=self._select_output_file
        ).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(files, text="Font (optional):").grid(
            row=3, column=0, sticky=W, padx=5, pady=5
        )
        ttk.Entry(files, textvariable=self.font_file, width=50).grid(
            row=3, column=1, sticky=(W, E), padx=5, pady=5
        )
        ttk.Button(
            files, text="Choose", command=self._select_font_file
        ).grid(row=3, column=2, padx=5, pady=5)

        params = ttk.LabelFrame(main, text="⚙️ Parameters", padding="10")
        params.grid(row=1, column=0, columnspan=2, sticky=(W, E), pady=5)

        size_frame = ttk.Frame(params)
        size_frame.grid(row=0, column=0, columnspan=2, sticky=W, pady=5)

        ttk.Label(size_frame, text="Page size:").pack(side=LEFT, padx=5)

        presets = ttk.Frame(size_frame)
        presets.pack(side=LEFT, padx=10)

        ttk.Button(
            presets,
            text="📱 Mobile (720x1280)",
            command=lambda: self._set_size(720, 1280),
        ).pack(side=LEFT, padx=2)
        ttk.Button(
            presets,
            text="📖 Book (1160x1534)",
            command=lambda: self._set_size(1160, 1534),
        ).pack(side=LEFT, padx=2)
        ttk.Button(
            presets,
            text="📺 HD (1920x1080)",
            command=lambda: self._set_size(1920, 1080),
        ).pack(side=LEFT, padx=2)
        ttk.Button(
            presets,
            text="▧ Square (1080x1080)",
            command=lambda: self._set_size(1080, 1080),
        ).pack(side=LEFT, padx=2)

        custom = ttk.Frame(params)
        custom.grid(row=1, column=0, columnspan=2, sticky=W, pady=5)

        ttk.Label(custom, text="Width:").pack(side=LEFT, padx=5)
        ttk.Spinbox(
            custom, from_=480, to=3840, textvariable=self.width, width=8
        ).pack(side=LEFT, padx=5)

        ttk.Label(custom, text="Height:").pack(side=LEFT, padx=5)
        ttk.Spinbox(
            custom, from_=360, to=2160, textvariable=self.height, width=8
        ).pack(side=LEFT, padx=5)

        ttk.Label(custom, text="FPS:").pack(side=LEFT, padx=15)
        ttk.Spinbox(
            custom, from_=15, to=60, textvariable=self.fps, width=5
        ).pack(side=LEFT, padx=5)

        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(W, E), pady=10)

        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, columnspan=2, pady=10)

        self.start_btn = ttk.Button(
            controls,
            text="Build",
            command=self._start_processing,
        )
        self.start_btn.pack(side=LEFT, padx=5)

        self.stop_btn = ttk.Button(
            controls, text="Stop", command=self._stop_processing, state=DISABLED
        )
        self.stop_btn.pack(side=LEFT, padx=5)

        log_frame = ttk.LabelFrame(main, text="📜 Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(W, E, N, S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)

        self.log_text = ScrolledText(log_frame, height=10, wrap=WORD)
        self.log_text.grid(row=0, column=0, sticky=(W, E, N, S))

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    # ------------------------------ UI helpers ------------------------------

    def _set_size(self, width: int, height: int) -> None:
        self.width.set(width)
        self.height.set(height)

    def _select_video_file(self) -> None:
        fname = filedialog.askopenfilename(
            title="Choose audio or video",
            filetypes=[
                (
                    "Media",
                    "*.mp4 *.avi *.mkv *.mov *.wav *.mp3 *.m4a *.aac *.flac *.ogg",
                ),
                ("Video", "*.mp4 *.avi *.mkv *.mov"),
                ("Audio", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg"),
                ("All", "*.*"),
            ],
        )
        if fname:
            self.video_file.set(fname)

    def _select_text_file(self) -> None:
        fname = filedialog.askopenfilename(
            title="Choose text file",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")],
        )
        if fname:
            self.text_file.set(fname)

    def _select_output_file(self) -> None:
        fname = filedialog.asksaveasfilename(
            title="Save video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All", "*.*")],
        )
        if fname:
            self.output_file.set(fname)

    def _select_font_file(self) -> None:
        fname = filedialog.askopenfilename(
            title="Choose font file",
            filetypes=[("Fonts", "*.ttf *.otf *.ttc"), ("All", "*.*")],
        )
        if fname:
            self.font_file.set(fname)

    def _log_gui(self, message: str) -> None:
        self.log_queue.put(message)

    def _progress_gui(self, value: int) -> None:
        self.progress["value"] = value

    def _update_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(END, msg + "\n")
                self.log_text.see(END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._update_logs)

    # ------------------------------ actions ---------------------------------

    def _start_processing(self) -> None:
        if self.processing:
            return

        if not self.video_file.get():
            messagebox.showerror("Error", "Choose an audio/video file.")
            return
        if not self.text_file.get():
            messagebox.showerror("Error", "Choose a text file.")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "Choose an output file name.")
            return

        self.processing = True
        self.start_btn["state"] = DISABLED
        self.stop_btn["state"] = NORMAL
        self.progress["value"] = 0
        self.log_text.delete(1.0, END)

        thread = threading.Thread(target=self._process_video, daemon=True)
        thread.start()

    def _process_video(self) -> None:
        try:
            self.current_creator = VideoBookCreator(
                video_file=self.video_file.get(),
                text_file=self.text_file.get(),
                output_video=self.output_file.get(),
                log_callback=self._log_gui,
                progress_callback=self._progress_gui,
                width=self.width.get(),
                height=self.height.get(),
                fps=self.fps.get(),
                font_path=self.font_file.get() or None,
            )
            self.current_creator.run()

            if not self.current_creator.stop_processing:
                self._log_gui("\n✨ Videobook created successfully!")
                messagebox.showinfo("Success", "Videobook created successfully!")
        except Exception as exc:
            self._log_gui(f"\n❌ ERROR: {exc}")
            messagebox.showerror("Error", f"An error occurred:\n{exc}")
        finally:
            self.processing = False
            self.current_creator = None
            self.start_btn["state"] = NORMAL
            self.stop_btn["state"] = DISABLED
            self.progress["value"] = 0

    def _stop_processing(self) -> None:
        confirm = messagebox.askyesno(
            "Confirm", "Are you sure you want to stop?"
        )
        if confirm:
            self._log_gui("\n⏹ Stopping...")
            if self.current_creator:
                self.current_creator.stop_processing = True
            self.processing = False
            self.start_btn["state"] = NORMAL
            self.stop_btn["state"] = DISABLED


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _main() -> int:
    if platform.system() == "Darwin":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ["MPLBACKEND"] = "Agg"
        LOGGER.info("🍎 macOS — SDL in headless mode")

    try:
        root = Tk()
        app = VideoBookApp(root)
        root.mainloop()
        return 0
    except Exception as exc:
        LOGGER.exception("Critical startup error: %s", exc)
        if platform.system() == "Darwin" and "SDLApplication" in str(exc):
            LOGGER.error("Detected SDL/Tkinter conflict on macOS.")
            LOGGER.error("Try:\npython -m tkinter videobook_ui.py\n"
                         "Or install imageio-ffmpeg.")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
