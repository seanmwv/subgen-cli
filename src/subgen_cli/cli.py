#!/usr/bin/env python3
"""Subgen CLI - Standalone command-line interface for subtitle generation

This is a clean, CLI-only implementation of subgen that uses Whisper AI
for transcription. It has no dependencies on server code and uses command-line
arguments instead of environment variables.

Based on subgen by McCloudS: https://github.com/McCloudS/subgen
"""

import argparse
import gc
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from enum import Enum
from io import BytesIO
from typing import Any

# Third-party imports
import ffmpeg
import stable_whisper
import torch

from . import __version__

# ============================================================================
# CONSTANTS
# ============================================================================

def get_default_model_path() -> str:
    """
    Get platform-specific default model cache directory.

    Returns:
        Path to the default model cache directory:
        - Windows: %LOCALAPPDATA%\\subgen\\models
        - Linux/macOS: ~/.cache/subgen/models
    """
    if os.name == 'nt':  # Windows
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home())))
        return str(base / "subgen" / "models")
    else:  # Linux/macOS
        base = Path(os.getenv('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        return str(base / 'subgen' / 'models')


VIDEO_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ogv",
    ".vob",
    ".rm",
    ".rmvb",
    ".ts",
    ".m4v",
    ".f4v",
    ".svq3",
    ".asf",
    ".m2ts",
    ".divx",
    ".xvid",
)

AUDIO_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
    ".alac",
    ".m4a",
    ".opus",
    ".aiff",
    ".aif",
    ".pcm",
    ".ra",
    ".ram",
    ".mid",
    ".midi",
    ".ape",
    ".wv",
    ".amr",
    ".vox",
    ".tak",
    ".spx",
    ".m4b",
    ".mka",
)


class LanguageCode(Enum):
    # ISO 639-1, ISO 639-2/T, ISO 639-2/B, English Name, Native Name
    AFAR = ("aa", "aar", "aar", "Afar", "Afar")
    AFRIKAANS = ("af", "afr", "afr", "Afrikaans", "Afrikaans")
    AMHARIC = ("am", "amh", "amh", "Amharic", "አማርኛ")
    ARABIC = ("ar", "ara", "ara", "Arabic", "العربية")
    ASSAMESE = ("as", "asm", "asm", "Assamese", "অসমীয়া")
    AZERBAIJANI = ("az", "aze", "aze", "Azerbaijani", "Azərbaycanca")
    BASHKIR = ("ba", "bak", "bak", "Bashkir", "Башҡортса")
    BELARUSIAN = ("be", "bel", "bel", "Belarusian", "Беларуская")
    BULGARIAN = ("bg", "bul", "bul", "Bulgarian", "Български")
    BENGALI = ("bn", "ben", "ben", "Bengali", "বাংলা")
    TIBETAN = ("bo", "bod", "tib", "Tibetan", "བོད་ཡིག")
    BRETON = ("br", "bre", "bre", "Breton", "Brezhoneg")
    BOSNIAN = ("bs", "bos", "bos", "Bosnian", "Bosanski")
    CATALAN = ("ca", "cat", "cat", "Catalan", "Català")
    CZECH = ("cs", "ces", "cze", "Czech", "Čeština")
    WELSH = ("cy", "cym", "wel", "Welsh", "Cymraeg")
    DANISH = ("da", "dan", "dan", "Danish", "Dansk")
    GERMAN = ("de", "deu", "ger", "German", "Deutsch")
    GREEK = ("el", "ell", "gre", "Greek", "Ελληνικά")
    ENGLISH = ("en", "eng", "eng", "English", "English")
    SPANISH = ("es", "spa", "spa", "Spanish", "Español")
    ESTONIAN = ("et", "est", "est", "Estonian", "Eesti")
    BASQUE = ("eu", "eus", "baq", "Basque", "Euskara")
    PERSIAN = ("fa", "fas", "per", "Persian", "فارسی")
    FINNISH = ("fi", "fin", "fin", "Finnish", "Suomi")
    FAROESE = ("fo", "fao", "fao", "Faroese", "Føroyskt")
    FRENCH = ("fr", "fra", "fre", "French", "Français")
    GALICIAN = ("gl", "glg", "glg", "Galician", "Galego")
    GUJARATI = ("gu", "guj", "guj", "Gujarati", "ગુજરાતી")
    HAUSA = ("ha", "hau", "hau", "Hausa", "Hausa")
    HAWAIIAN = ("haw", "haw", "haw", "Hawaiian", "ʻŌlelo Hawaiʻi")
    HEBREW = ("he", "heb", "heb", "Hebrew", "עברית")
    HINDI = ("hi", "hin", "hin", "Hindi", "हिन्दी")
    CROATIAN = ("hr", "hrv", "hrv", "Croatian", "Hrvatski")
    HAITIAN_CREOLE = ("ht", "hat", "hat", "Haitian Creole", "Kreyòl Ayisyen")
    HUNGARIAN = ("hu", "hun", "hun", "Hungarian", "Magyar")
    ARMENIAN = ("hy", "hye", "arm", "Armenian", "Հայերեն")
    INDONESIAN = ("id", "ind", "ind", "Indonesian", "Bahasa Indonesia")
    ICELANDIC = ("is", "isl", "ice", "Icelandic", "Íslenska")
    ITALIAN = ("it", "ita", "ita", "Italian", "Italiano")
    JAPANESE = ("ja", "jpn", "jpn", "Japanese", "日本語")
    JAVANESE = ("jw", "jav", "jav", "Javanese", "ꦧꦱꦗꦮ")
    GEORGIAN = ("ka", "kat", "geo", "Georgian", "ქართული")
    KAZAKH = ("kk", "kaz", "kaz", "Kazakh", "Қазақша")
    KHMER = ("km", "khm", "khm", "Khmer", "ភាសាខ្មែរ")
    KANNADA = ("kn", "kan", "kan", "Kannada", "ಕನ್ನಡ")
    KOREAN = ("ko", "kor", "kor", "Korean", "한국어")
    LATIN = ("la", "lat", "lat", "Latin", "Latina")
    LUXEMBOURGISH = ("lb", "ltz", "ltz", "Luxembourgish", "Lëtzebuergesch")
    LINGALA = ("ln", "lin", "lin", "Lingala", "Lingála")
    LAO = ("lo", "lao", "lao", "Lao", "ພາສາລາວ")
    LITHUANIAN = ("lt", "lit", "lit", "Lithuanian", "Lietuvių")
    LATVIAN = ("lv", "lav", "lav", "Latvian", "Latviešu")
    MALAGASY = ("mg", "mlg", "mlg", "Malagasy", "Malagasy")
    MAORI = ("mi", "mri", "mao", "Maori", "Te Reo Māori")
    MACEDONIAN = ("mk", "mkd", "mac", "Macedonian", "Македонски")
    MALAYALAM = ("ml", "mal", "mal", "Malayalam", "മലയാളം")
    MONGOLIAN = ("mn", "mon", "mon", "Mongolian", "Монгол")
    MARATHI = ("mr", "mar", "mar", "Marathi", "मराठी")
    MALAY = ("ms", "msa", "may", "Malay", "Bahasa Melayu")
    MALTESE = ("mt", "mlt", "mlt", "Maltese", "Malti")
    BURMESE = ("my", "mya", "bur", "Burmese", "မြန်မာစာ")
    NEPALI = ("ne", "nep", "nep", "Nepali", "नेपाली")
    DUTCH = ("nl", "nld", "dut", "Dutch", "Nederlands")
    NORWEGIAN_NYNORSK = ("nn", "nno", "nno", "Norwegian Nynorsk", "Nynorsk")
    NORWEGIAN = ("no", "nor", "nor", "Norwegian", "Norsk")
    OCCITAN = ("oc", "oci", "oci", "Occitan", "Occitan")
    PUNJABI = ("pa", "pan", "pan", "Punjabi", "ਪੰਜਾਬੀ")
    POLISH = ("pl", "pol", "pol", "Polish", "Polski")
    PASHTO = ("ps", "pus", "pus", "Pashto", "پښتو")
    PORTUGUESE = ("pt", "por", "por", "Portuguese", "Português")
    ROMANIAN = ("ro", "ron", "rum", "Romanian", "Română")
    RUSSIAN = ("ru", "rus", "rus", "Russian", "Русский")
    SANSKRIT = ("sa", "san", "san", "Sanskrit", "संस्कृतम्")
    SINDHI = ("sd", "snd", "snd", "Sindhi", "سنڌي")
    SINHALA = ("si", "sin", "sin", "Sinhala", "සිංහල")
    SLOVAK = ("sk", "slk", "slo", "Slovak", "Slovenčina")
    SLOVENE = ("sl", "slv", "slv", "Slovene", "Slovenščina")
    SHONA = ("sn", "sna", "sna", "Shona", "ChiShona")
    SOMALI = ("so", "som", "som", "Somali", "Soomaaliga")
    ALBANIAN = ("sq", "sqi", "alb", "Albanian", "Shqip")
    SERBIAN = ("sr", "srp", "srp", "Serbian", "Српски")
    SUNDANESE = ("su", "sun", "sun", "Sundanese", "Basa Sunda")
    SWEDISH = ("sv", "swe", "swe", "Swedish", "Svenska")
    SWAHILI = ("sw", "swa", "swa", "Swahili", "Kiswahili")
    TAMIL = ("ta", "tam", "tam", "Tamil", "தமிழ்")
    TELUGU = ("te", "tel", "tel", "Telugu", "తెలుగు")
    TAJIK = ("tg", "tgk", "tgk", "Tajik", "Тоҷикӣ")
    THAI = ("th", "tha", "tha", "Thai", "ไทย")
    TURKMEN = ("tk", "tuk", "tuk", "Turkmen", "Türkmençe")
    TAGALOG = ("tl", "tgl", "tgl", "Tagalog", "Tagalog")
    TURKISH = ("tr", "tur", "tur", "Turkish", "Türkçe")
    TATAR = ("tt", "tat", "tat", "Tatar", "Татарча")
    UKRAINIAN = ("uk", "ukr", "ukr", "Ukrainian", "Українська")
    URDU = ("ur", "urd", "urd", "Urdu", "اردو")
    UZBEK = ("uz", "uzb", "uzb", "Uzbek", "Oʻzbek")
    VIETNAMESE = ("vi", "vie", "vie", "Vietnamese", "Tiếng Việt")
    YIDDISH = ("yi", "yid", "yid", "Yiddish", "ייִדיש")
    YORUBA = ("yo", "yor", "yor", "Yoruba", "Yorùbá")
    CHINESE = ("zh", "zho", "chi", "Chinese", "中文")
    CANTONESE = ("yue", "yue", "yue", "Cantonese", "粵語")
    NONE = (None, None, None, None, None)  # For no language
    # und for Undetermined aka unknown language https://www.loc.gov/standards/iso639-2/faq.html#25

    def __init__(self, iso_639_1, iso_639_2_t, iso_639_2_b, name_en, name_native):
        self.iso_639_1 = iso_639_1
        self.iso_639_2_t = iso_639_2_t
        self.iso_639_2_b = iso_639_2_b
        self.name_en = name_en
        self.name_native = name_native

    @staticmethod
    def from_iso_639_1(code):
        for lang in LanguageCode:
            if lang.iso_639_1 == code:
                return lang
        return LanguageCode.NONE

    @staticmethod
    def from_iso_639_2(code):
        for lang in LanguageCode:
            if lang.iso_639_2_t == code or lang.iso_639_2_b == code:
                return lang
        return LanguageCode.NONE

    @staticmethod
    def from_name(name: str):
        """Convert a language name (either English or native) to LanguageCode enum."""
        for lang in LanguageCode:
            if (
                lang.name_en.lower() == name.lower()
                or lang.name_native.lower() == name.lower()
            ):
                return lang
        return LanguageCode.NONE

    @staticmethod
    def from_string(value: str):
        """Convert a string to a LanguageCode instance. Matches on ISO codes, English name, or native name.
        """
        if value is None:
            return LanguageCode.NONE
        value = value.strip().lower()
        for lang in LanguageCode:
            if lang is LanguageCode.NONE:
                continue
            if (
                value == lang.iso_639_1
                or value == lang.iso_639_2_t
                or value == lang.iso_639_2_b
                or value == lang.name_en.lower()
                or value == lang.name_native.lower()
            ):
                return lang
        return LanguageCode.NONE

    # is valid language
    @staticmethod
    def is_valid_language(language: str):
        return LanguageCode.from_string(language) is not LanguageCode.NONE

    def to_iso_639_1(self):
        return self.iso_639_1

    def to_iso_639_2_t(self):
        return self.iso_639_2_t

    def to_iso_639_2_b(self):
        return self.iso_639_2_b

    def to_name(self, in_english=True):
        return self.name_en if in_english else self.name_native

    def __str__(self):
        if self.name_en is None:
            return "Unknown"
        return self.name_en

    def __bool__(self):
        return True if self.iso_639_1 is not None else False

    def __eq__(self, other):
        """Compare the LanguageCode instance to another object.
        Explicitly handle comparison to None.
        """
        if other is None:
            # If compared to None, return False unless self is None
            return self.iso_639_1 is None
        if isinstance(other, str):  # Allow comparison with a string
            return self.value == LanguageCode.from_string(other)
        if isinstance(other, LanguageCode):
            # Normal comparison for LanguageCode instances
            return self.iso_639_1 == other.iso_639_1
        # Otherwise, defer to the default equality
        return NotImplemented


# ============================================================================
# WHISPER MODEL MANAGER
# ============================================================================


class WhisperModelManager:
    """Manages the lifecycle of a Whisper model without global state.

    This class handles loading, using, and cleaning up Whisper models,
    replacing the global model pattern from the original subgen.py.
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "auto",
        threads: int = 4,
        model_path: str = None,
    ):
        """Initialize the model manager.

        Args:
            model_name: Whisper model size (tiny/base/small/medium/large)
            device: Compute device (cpu/cuda/gpu)
            compute_type: Quantization type (auto/int8/float16/etc.)
            threads: Number of CPU threads to use
            model_path: Directory to store/load models (defaults to platform-specific cache)

        """
        self.model_name = model_name
        self.device = "cuda" if device.lower() in ("gpu", "cuda") else "cpu"
        self.compute_type = compute_type
        self.threads = threads
        self.model_path = model_path if model_path is not None else get_default_model_path()
        self._model = None

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

    def load(self):
        """Load the Whisper model into memory."""
        if self._model is None:
            logging.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self._model = stable_whisper.load_faster_whisper(
                self.model_name,
                download_root=self.model_path,
                device=self.device,
                cpu_threads=self.threads,
                num_workers=1,
                compute_type=self.compute_type,
            )
            logging.debug("Model loaded successfully")

    def unload(self):
        """Unload the model from memory and clean up resources."""
        if self._model is not None:
            logging.debug("Unloading model from memory")
            self._model.model.unload_model()
            del self._model
            self._model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared")

            # Garbage collection (skip on Windows to avoid crashes)
            if os.name != "nt":
                gc.collect()

    @property
    def model(self):
        """Get the model, loading it if necessary."""
        if self._model is None:
            self.load()
        return self._model

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None


@contextmanager
def whisper_model_context(
    model_name: str = "medium",
    device: str = "cpu",
    compute_type: str = "auto",
    threads: int = 4,
    model_path: str = None,
):
    """Context manager for Whisper model lifecycle.

    Ensures the model is properly loaded and cleaned up, even if errors occur.
    Perfect for CLI usage where we load once, transcribe, and exit.

    Usage:
        with whisper_model_context("medium", "cpu") as model_manager:
            result = model_manager.model.transcribe(...)
    """
    manager = WhisperModelManager(model_name, device, compute_type, threads, model_path)
    try:
        manager.load()
        yield manager
    finally:
        manager.unload()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def is_audio_file(file_path: str) -> bool:
    """Check if a file is an audio file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in AUDIO_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """Check if a file is a video file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def is_media_file(file_path: str) -> bool:
    """Check if a file is a supported media file."""
    return is_audio_file(file_path) or is_video_file(file_path)


def progress_callback(seek, total):
    """Progress callback for Whisper transcription."""
    sys.stdout.flush()
    sys.stderr.flush()
    # Simple progress indicator - can be enhanced


def get_audio_tracks(video_file: str) -> list[dict[str, Any]]:
    """Extract information about audio tracks in a media file.

    Args:
        video_file: Path to the media file

    Returns:
        List of dictionaries with audio track information:
        - index (int): Stream index
        - codec (str): Audio codec name
        - channels (int): Number of audio channels
        - language (LanguageCode): Track language
        - title (str): Track title
        - default (bool): Is default track
        - forced (bool): Is forced track
        - original (bool): Is original track
        - commentary (bool): Is commentary track

    """
    try:
        probe = ffmpeg.probe(video_file, select_streams="a")
        audio_streams = probe.get("streams", [])

        audio_tracks = []
        for stream in audio_streams:
            audio_track = {
                "index": int(stream.get("index", 0)),
                "codec": stream.get("codec_name", "Unknown"),
                "channels": int(stream.get("channels", 0)),
                "language": LanguageCode.from_iso_639_2(
                    stream.get("tags", {}).get("language", "Unknown"),
                ),
                "title": stream.get("tags", {}).get("title", "None"),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                "original": stream.get("disposition", {}).get("original", 0) == 1,
                "commentary": "commentary"
                in stream.get("tags", {}).get("title", "").lower(),
            }
            audio_tracks.append(audio_track)

        return audio_tracks

    except ffmpeg.Error as e:
        logging.exception(f"FFmpeg error reading audio tracks: {e.stderr}")
        return []
    except Exception as e:
        logging.exception(f"Error reading audio tracks: {e}")
        return []


def extract_audio_track_to_memory(
    video_path: str, track_index: int,
) -> BytesIO | None:
    """Extract a specific audio track from a video file to memory.

    Args:
        video_path: Path to the video file
        track_index: Index of the audio track to extract

    Returns:
        BytesIO object containing audio data, or None if extraction failed

    """
    try:
        out, _ = (
            ffmpeg.input(video_path)
            .output(
                "pipe:",
                map=f"0:{track_index}",
                format="wav",
                ac=1,  # Mono audio
                ar=16000,  # 16 kHz sample rate (optimal for speech)
                loglevel="quiet",
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        return BytesIO(out)

    except ffmpeg.Error as e:
        logging.exception(f"FFmpeg error extracting audio track: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.exception(f"Error extracting audio track: {e}")
        return None


def get_audio_track_by_language(
    audio_tracks: list[dict[str, Any]], language: LanguageCode,
) -> dict[str, Any] | None:
    """Find the first audio track with the specified language.

    Args:
        audio_tracks: List of audio track dictionaries
        language: Language to search for

    Returns:
        Audio track dictionary, or None if not found

    """
    for track in audio_tracks:
        if track["language"] == language:
            return track
    return None


def handle_multiple_audio_tracks(
    file_path: str, language: LanguageCode | None = None,
) -> BytesIO | None:
    """Handle media files with multiple audio tracks.

    If the file has multiple audio tracks, extract the one matching
    the specified language, or the first track if no language specified.

    Args:
        file_path: Path to the media file
        language: Preferred language for audio track

    Returns:
        BytesIO with extracted audio, or None if not extracted

    """
    audio_tracks = get_audio_tracks(file_path)

    if len(audio_tracks) <= 1:
        return None

    logging.debug(f"File has {len(audio_tracks)} audio tracks")
    logging.debug(
        "Audio tracks:\n"
        + "\n".join(
            [
                f"  - {track['index']}: {track['codec']} {track['language'].to_name()} "
                f"{'[DEFAULT]' if track['default'] else ''}"
                for track in audio_tracks
            ],
        ),
    )

    # Select track by language if specified
    audio_track = None
    if language is not None and language != LanguageCode.NONE:
        audio_track = get_audio_track_by_language(audio_tracks, language)
        if audio_track:
            logging.debug(f"Selected track by language: {language.to_name()}")

    # Fall back to first track
    if audio_track is None:
        audio_track = audio_tracks[0]
        logging.debug(f"Using first audio track (index {audio_track['index']})")

    # Extract the selected track
    audio_bytes = extract_audio_track_to_memory(file_path, audio_track["index"])
    if audio_bytes is None:
        logging.error(f"Failed to extract audio track {audio_track['index']}")

    return audio_bytes


def write_lrc(result, file_path: str):
    """Write transcription result to LRC (lyrics) format.

    Args:
        result: Whisper transcription result
        file_path: Output path for LRC file

    """
    with open(file_path, "w", encoding="utf-8") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            # Remove embedded newlines (some players ignore text after newlines)
            text = segment.text.replace("\n", "")
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}]{text}\n")


def build_subtitle_filename(
    base_path: str,
    language: LanguageCode,
    model_name: str,
    is_lrc: bool = False,
    include_model: bool = True,
    include_subgen: bool = True,
    subtitle_name_override: str | None = None,
) -> str:
    """Build the subtitle filename based on configuration.

    Args:
        base_path: Base file path (without extension)
        language: Subtitle language
        model_name: Whisper model name
        is_lrc: Whether this is an LRC file (vs SRT)
        include_model: Include model name in filename
        include_subgen: Include 'subgen' in filename
        subtitle_name_override: Override for language part of filename

    Returns:
        Full path for subtitle file

    """
    subgen_part = ".subgen" if include_subgen else ""
    model_part = f".{model_name}" if include_model else ""

    if is_lrc:
        # LRC format: basename.subgen.model.lrc
        return f"{base_path}{subgen_part}{model_part}.lrc"
    # SRT format: basename.subgen.model.lang.srt
    lang_part = subtitle_name_override or language.to_iso_639_2_b()
    return f"{base_path}{subgen_part}{model_part}.{lang_part}.srt"


# ============================================================================
# TRANSCRIPTION
# ============================================================================


class TranscriptionConfig:
    """Configuration for transcription operation."""

    def __init__(
        self,
        # Model settings
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "auto",
        threads: int = 4,
        model_path: str = None,
        # Transcription settings
        task: str = "transcribe",
        language: LanguageCode | None = None,
        audio_track: int | None = None,
        # Output settings
        output_dir: str | None = None,
        word_highlight: bool = False,
        lrc_for_audio: bool = True,
        include_model_name: bool = True,
        include_subgen_tag: bool = True,
        subtitle_name: str | None = None,
        # Advanced settings
        custom_regroup: str = "cm_sl=84_sl=42++++++1",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.threads = threads
        self.model_path = model_path if model_path is not None else get_default_model_path()

        self.task = task
        self.language = language
        self.audio_track = audio_track

        self.output_dir = output_dir
        self.word_highlight = word_highlight
        self.lrc_for_audio = lrc_for_audio
        self.include_model_name = include_model_name
        self.include_subgen_tag = include_subgen_tag
        self.subtitle_name = subtitle_name

        self.custom_regroup = custom_regroup
        self.verbose = verbose


def transcribe_file(file_path: str, config: TranscriptionConfig) -> str:
    """Transcribe a media file to generate subtitles.

    Args:
        file_path: Path to the media file
        config: Transcription configuration

    Returns:
        Path to the generated subtitle file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file is not a supported media type
        Exception: For transcription errors

    """
    # Validate file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_media_file(file_path):
        ext = os.path.splitext(file_path)[1]
        raise ValueError(f"Unsupported file type: {ext}")

    logging.info(f"Processing: {os.path.basename(file_path)}")

    # Determine if this is an audio file
    is_audio = is_audio_file(file_path)

    # Start timing
    start_time = time.time()

    # Use context manager for model lifecycle
    with whisper_model_context(
        config.model_name,
        config.device,
        config.compute_type,
        config.threads,
        config.model_path,
    ) as model_manager:

        # Handle audio track selection
        data = file_path
        if config.audio_track is not None:
            # User specified a specific track (1-based index)
            # Need to convert to actual FFmpeg stream index
            tracks = get_audio_tracks(file_path)

            if not tracks:
                raise ValueError("No audio tracks found in file")

            if config.audio_track < 1 or config.audio_track > len(tracks):
                raise ValueError(
                    f"Audio track {config.audio_track} not found. "
                    f"Valid tracks: 1-{len(tracks)}. Use --list-tracks to see available tracks.",
                )

            # Convert 1-based user input to actual stream index
            selected_track = tracks[config.audio_track - 1]
            stream_index = selected_track["index"]

            logging.info(
                f"Extracting track {config.audio_track} "
                f"(stream {stream_index}, {selected_track['codec']}, "
                f"{selected_track['language'].to_name()})",
            )

            extracted = extract_audio_track_to_memory(file_path, stream_index)
            if extracted:
                data = extracted.read()
            else:
                raise Exception(
                    f"Failed to extract audio track {config.audio_track} (stream {stream_index})",
                )
        elif not is_audio:
            # Handle multiple audio tracks automatically
            extracted = handle_multiple_audio_tracks(file_path, config.language)
            if extracted:
                data = extracted.read()

        # Prepare transcription arguments
        transcribe_args = {"progress_callback": progress_callback}

        if config.custom_regroup:
            transcribe_args["regroup"] = config.custom_regroup

        # Determine language parameter
        lang_param = (
            config.language.to_iso_639_1()
            if config.language and config.language != LanguageCode.NONE
            else None
        )

        # Transcribe
        logging.info(f"Starting {config.task}...")
        if config.verbose:
            logging.info(
                f"Model: {config.model_name}, Device: {config.device}, Compute: {config.compute_type}",
            )

        result = model_manager.model.transcribe(
            data, language=lang_param, task=config.task, **transcribe_args,
        )

        # Determine detected language
        detected_language = config.language
        if not detected_language or detected_language == LanguageCode.NONE:
            detected_language = LanguageCode.from_string(result.language)

        logging.info(f"Detected language: {detected_language.to_name()}")

        # Build output path
        file_name, file_extension = os.path.splitext(file_path)

        if config.output_dir:
            # Custom output directory
            os.makedirs(config.output_dir, exist_ok=True)
            base_name = os.path.basename(file_name)
            base_path = os.path.join(config.output_dir, base_name)
        else:
            # Same directory as input
            base_path = file_name

        # Generate subtitle file
        if is_audio and config.lrc_for_audio:
            # Generate LRC for audio files
            output_file = build_subtitle_filename(
                base_path,
                detected_language,
                config.model_name,
                is_lrc=True,
                include_model=config.include_model_name,
                include_subgen=config.include_subgen_tag,
                subtitle_name_override=config.subtitle_name,
            )
            write_lrc(result, output_file)
        else:
            # Generate SRT for video files
            output_file = build_subtitle_filename(
                base_path,
                detected_language,
                config.model_name,
                is_lrc=False,
                include_model=config.include_model_name,
                include_subgen=config.include_subgen_tag,
                subtitle_name_override=config.subtitle_name,
            )
            result.to_srt_vtt(output_file, word_level=config.word_highlight)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)

        logging.info(f"Completed in {minutes}m {seconds}s")
        logging.info(f"Subtitle saved: {output_file}")

        return output_file


# ============================================================================
# CLI INTERFACE
# ============================================================================


def setup_logging(verbose: bool = False):
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - %(message)s" if verbose else "%(message)s"
        ),
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def list_audio_tracks(file_path: str):
    """List audio tracks in a media file."""
    tracks = get_audio_tracks(file_path)
    if not tracks:
        logging.warning("No audio tracks found in file")
        return

    logging.info(f"\nAudio tracks found ({len(tracks)}):")
    for idx, track in enumerate(tracks, 1):
        lang = (
            track["language"].to_name()
            if track["language"] != LanguageCode.NONE
            else "Unknown"
        )
        default = " [DEFAULT]" if track.get("default", False) else ""
        logging.info(f"  Track {idx}: {track['codec']} - {lang}{default}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles for video/audio files using Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f video.mkv
  %(prog)s -f movie.mp4 -l es -a 2
  %(prog)s -f anime.mkv -t translate -w
  %(prog)s -f video.mp4 -o ./subtitles -n en-custom
  %(prog)s -f audio.mp3 --no-lrc

Model Options:
  Use --model to specify Whisper model size:
    - tiny (fastest, least accurate)
    - base
    - small
    - medium (default, balanced)
    - large (slowest, most accurate)

Device Options:
  Use --device to specify compute device:
    - cpu (default, works everywhere)
    - cuda/gpu (requires NVIDIA GPU with CUDA)

For more information: https://github.com/seanmwv/subgen-cli
        """,
    )

    # Required arguments
    parser.add_argument(
        "-f", "--file", required=True, help="Path to video or audio file",
    )

    # Model configuration
    model_group = parser.add_argument_group("model configuration")
    model_group.add_argument(
        "--model",
        default="medium",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "distil-large-v2",
            "distil-large-v3",
        ],
        help="Whisper model size (default: medium)",
    )
    model_group.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "gpu"],
        help="Compute device (default: cpu)",
    )
    model_group.add_argument(
        "--compute-type",
        default="auto",
        help="Quantization type: auto, int8, float16, etc. (default: auto)",
    )
    model_group.add_argument(
        "--threads", type=int, default=4, help="Number of CPU threads (default: 4)",
    )
    model_group.add_argument(
        "--model-path",
        default=None,
        help=f"Directory to store/load models (default: {get_default_model_path()})",
    )

    # Transcription options
    trans_group = parser.add_argument_group("transcription options")
    trans_group.add_argument(
        "-l",
        "--language",
        help="Force language (2-letter ISO 639-1 code, e.g., en, es, fr). If not set, language will be auto-detected.",
    )
    trans_group.add_argument(
        "-t",
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: transcribe (same language) or translate (to English). Default: transcribe",
    )
    trans_group.add_argument(
        "-a",
        "--audio-track",
        type=int,
        help="Audio track number to use (1-based index). Use --list-tracks to see available tracks.",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-o",
        "--output",
        help="Output directory for subtitle file (default: same directory as input file)",
    )
    output_group.add_argument(
        "-w",
        "--word-highlight",
        action="store_true",
        help="Enable word-level highlighting (karaoke-style subtitles)",
    )
    output_group.add_argument(
        "-n",
        "--subtitle-name",
        help="Custom subtitle language name (overrides auto-detected language)",
    )
    output_group.add_argument(
        "--no-lrc",
        action="store_true",
        help="Generate SRT for audio files instead of LRC",
    )
    output_group.add_argument(
        "--no-model-name",
        action="store_true",
        help="Do not include model name in subtitle filename",
    )
    output_group.add_argument(
        "--no-subgen-tag",
        action="store_true",
        help='Do not include "subgen" tag in subtitle filename',
    )

    # Information options
    info_group = parser.add_argument_group("information options")
    info_group.add_argument(
        "--list-tracks",
        action="store_true",
        help="List available audio tracks and exit",
    )
    info_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging",
    )
    info_group.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("advanced options")
    advanced_group.add_argument(
        "--regroup",
        default="cm_sl=84_sl=42++++++1",
        help="Custom regroup pattern for subtitle formatting (default: cm_sl=84_sl=42++++++1)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Validate file
    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        logging.error(f"Error: File not found: {file_path}")
        sys.exit(2)

    # Handle --list-tracks
    if args.list_tracks:
        list_audio_tracks(file_path)
        sys.exit(0)

    # Parse language if specified
    force_language = LanguageCode.NONE
    if args.language:
        try:
            force_language = LanguageCode.from_string(args.language)
            logging.info(
                f"Forcing language: {force_language.to_name()} ({args.language})",
            )
        except (ValueError, AttributeError):
            logging.exception(f"Error: Invalid language code: {args.language}")
            logging.exception("Use 2-letter ISO 639-1 codes (e.g., 'en', 'es', 'fr')")
            sys.exit(1)

    # Build configuration
    config = TranscriptionConfig(
        # Model settings
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        threads=args.threads,
        model_path=args.model_path,
        # Transcription settings
        task=args.task,
        language=force_language if force_language != LanguageCode.NONE else None,
        audio_track=args.audio_track,
        # Output settings
        output_dir=args.output,
        word_highlight=args.word_highlight,
        lrc_for_audio=not args.no_lrc,
        include_model_name=not args.no_model_name,
        include_subgen_tag=not args.no_subgen_tag,
        subtitle_name=args.subtitle_name,
        # Advanced settings
        custom_regroup=args.regroup,
        verbose=args.verbose,
    )

    # Show configuration in verbose mode
    if args.verbose:
        logging.info(f"Subgen CLI v{__version__}")
        logging.info(f"Model: {config.model_name}")
        logging.info(f"Device: {config.device}")
        logging.info(f"Compute type: {config.compute_type}")
        logging.info(f"Threads: {config.threads}")

    # Transcribe the file
    try:
        output_file = transcribe_file(file_path, config)
        print(output_file)  # Print path for scripting purposes
        sys.exit(0)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        if config.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
