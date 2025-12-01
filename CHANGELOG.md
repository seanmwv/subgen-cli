# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-30 (YYYY-MM-DD)

### Added
- Initial release of Subgen CLI
- Standalone command-line interface for subtitle generation
- Whisper AI integration for accurate transcription
- Support for multiple Whisper models (tiny, base, small, medium, large, large-v2, large-v3, distil-large-v2, distil-large-v3)
- 100+ language support with automatic detection
- Multi-track audio file support
- CPU and CUDA/GPU device support
- SRT subtitle format for video files
- LRC subtitle format for audio files
- Word-level highlighting (karaoke-style subtitles)
- Translation capability (transcribe to English)
- Audio track selection for multi-track files
- Customizable output directory
- Configurable subtitle file naming
- Verbose logging mode
- Audio track listing functionality
- Command-line argument parsing
- ISO 639-1 and ISO 639-2 language code support
- Native language name support
- Platform-specific model cache directories (Windows: %LOCALAPPDATA%\subgen\models, Linux/macOS: ~/.cache/subgen/models)

### Features
- Clean, CLI-only implementation (no server dependencies)
- Command-line arguments instead of environment variables
- Simplified model lifecycle management
- Context manager for automatic model cleanup
- Comprehensive language code conversion system
- Support for both video and audio file formats
- Automatic audio track extraction for multi-track files
- Configurable subtitle regrouping patterns
- Custom compute type selection (auto, int8, float16)
- Thread count configuration for CPU usage
- Model path customization for offline usage

### Technical Details
- Based on [McCloudS/subgen](https://github.com/McCloudS/subgen)
- Built with Python 3.11+
- Uses stable-ts for improved timestamp stability
- FFmpeg integration for audio/video processing
- PyTorch for model inference
- MIT License

[Unreleased]: https://github.com/seanmwv/subgen-cli/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/seanmwv/subgen-cli/releases/tag/v1.0.0
