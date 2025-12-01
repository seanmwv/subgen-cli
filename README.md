# Subgen CLI

A standalone command-line interface for automated subtitle generation using OpenAI's Whisper AI. This is a clean, CLI-focused implementation that generates high-quality subtitles for video and audio files.

## Features

- **Automatic Speech Recognition**: Uses Whisper AI for accurate transcription
- **Multiple Whisper Models**: Support for tiny, base, small, medium, and large models
- **100+ Languages**: Comprehensive language support with automatic detection
- **Multi-track Support**: Handle files with multiple audio tracks
- **CPU & GPU**: Works on CPU or NVIDIA GPUs with CUDA
- **Flexible Output**:
  - SRT format for videos
  - LRC format for audio files
  - Word-level highlighting (karaoke-style)
- **Translation**: Transcribe in original language or translate to English
- **Customizable**: Control output naming, directory, and subtitle formatting

## Installation

### Via pip (recommended)

```bash
pip install subgen-cli
```

### From source

```bash
git clone https://github.com/seanmwv/subgen-cli.git
cd subgen-cli
pip install -e .
```

## Requirements

- **Python**: 3.11 or higher
- **FFmpeg**: Required for audio/video processing
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo yum install ffmpeg` (RHEL/CentOS)

### Optional: GPU Support

For NVIDIA GPU acceleration:

```bash
pip install subgen-cli[gpu]
```

Requires CUDA-compatible GPU and drivers.

## Usage

### Basic Usage

Generate subtitles with automatic language detection:

```bash
subgen -f video.mp4
```

### Common Examples

```bash
# Specify language (Spanish)
subgen -f movie.mp4 -l es

# Use specific audio track
subgen -f anime.mkv -a 2

# Translate to English
subgen -f video.mp4 -t translate

# Use faster model
subgen -f video.mp4 --model small

# Save to specific directory
subgen -f video.mp4 -o ./subtitles

# Enable word-level highlighting (karaoke style)
subgen -f video.mp4 -w

# List available audio tracks
subgen -f video.mkv --list-tracks

# Use GPU for faster processing
subgen -f video.mp4 --device cuda --model large
```

### Command-Line Options

#### Required Arguments

- `-f, --file`: Path to video or audio file

#### Model Configuration

- `--model`: Whisper model size (default: medium)
  - `tiny`: Fastest, least accurate
  - `base`: Fast, basic accuracy
  - `small`: Balanced speed and accuracy
  - `medium`: **Recommended** - Good balance (default)
  - `large`, `large-v2`, `large-v3`: Slowest, most accurate
  - `distil-large-v2`, `distil-large-v3`: Faster alternatives to large models
- `--device`: Compute device (default: cpu)
  - `cpu`: Works everywhere
  - `cuda` or `gpu`: Requires NVIDIA GPU with CUDA
- `--compute-type`: Quantization type (default: auto)
  - Options: `auto`, `int8`, `float16`
- `--threads`: Number of CPU threads (default: 4)
- `--model-path`: Directory to store/load models
  - **Windows**: Default: `%LOCALAPPDATA%\subgen\models`
  - **Linux/macOS**: Default: `~/.cache/subgen/models`
  - Models are cached globally and shared across all projects

#### Transcription Options

- `-l, --language`: Force specific language (2-letter ISO 639-1 code)
  - Examples: `en` (English), `es` (Spanish), `fr` (French), `ja` (Japanese)
  - If not specified, language is auto-detected
- `-t, --task`: Task to perform (default: transcribe)
  - `transcribe`: Keep original language
  - `translate`: Translate to English
- `-a, --audio-track`: Select specific audio track by number (1-based index)

#### Output Options

- `-o, --output`: Output directory for subtitle file
- `-w, --word-highlight`: Enable word-level highlighting (karaoke-style)
- `-n, --subtitle-name`: Custom subtitle language name
- `--no-lrc`: Generate SRT for audio files instead of LRC
- `--no-model-name`: Exclude model name from filename
- `--no-subgen-tag`: Exclude "subgen" tag from filename

#### Information Options

- `--list-tracks`: List available audio tracks and exit
- `-v, --verbose`: Enable verbose logging
- `--version`: Show version and exit

#### Advanced Options

- `--regroup`: Custom regroup pattern for subtitle formatting
  - Default: `cm_sl=84_sl=42++++++1`

## Configuration

Subgen CLI uses command-line arguments for all configuration. There are no environment variables or configuration files.

### Output File Naming

By default, subtitle files are named with the following pattern:

```
[original_name].subgen.[model].[language].srt
```

Examples:
- `movie.subgen.medium.eng.srt`
- `audio.subgen.small.lrc`

You can customize this with:
- `--no-subgen-tag`: Remove "subgen" from filename
- `--no-model-name`: Remove model name from filename
- `-n custom`: Override language code with custom name

### Supported Languages

Subgen supports 100+ languages through Whisper AI, including:

Arabic, Chinese (Mandarin & Cantonese), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese, and many more.

Use 2-letter ISO 639-1 codes with the `-l` flag (e.g., `-l fr` for French).

## Performance Tips

1. **Model Selection**:
   - Use `tiny` or `small` for quick processing
   - Use `medium` for balanced quality (recommended)
   - Use `large` for maximum accuracy (slower)

2. **GPU Acceleration**:
   - Install GPU support: `pip install subgen-cli[gpu]`
   - Use `--device cuda` for 5-10x faster processing

3. **Multi-track Files**:
   - Use `--list-tracks` to see available audio tracks
   - Select the correct track with `-a` to avoid processing wrong audio

## Troubleshooting

### FFmpeg Not Found

If you get an error about FFmpeg not being found:

1. Install FFmpeg (see Requirements section)
2. Ensure FFmpeg is in your system PATH
3. Verify installation: `ffmpeg -version`

### GPU Not Working (Windows)

GPU device selection may not work correctly on Windows due to path handling. Use CPU mode if you encounter issues.

### Out of Memory

If you run out of memory:

1. Use a smaller model (`--model small` or `--model tiny`)
2. Reduce thread count (`--threads 2`)
3. Close other applications

### Wrong Language Detected

If the wrong language is detected:

1. Specify language explicitly: `-l en` (or appropriate code)
2. Check audio quality - poor audio affects detection
3. Try a larger model for better accuracy

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/seanmwv/subgen-cli.git
cd subgen-cli

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
ruff check src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Attribution

This project is based on [McCloudS/subgen](https://github.com/McCloudS/subgen), a subtitle generation tool originally designed for server environments. Subgen CLI is a standalone, command-line focused standalone reimplementation without environment variables.

Original work Copyright (c) 2023 McCloudS
Modified work Copyright (c) 2025 seanmwv

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [stable-ts](https://github.com/jianfch/stable-ts) for improved timestamp stability
- [McCloudS](https://github.com/McCloudS) for the original subgen implementation
