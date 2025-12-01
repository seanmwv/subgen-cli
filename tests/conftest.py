"""Pytest configuration and shared fixtures"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a mock video file for testing"""
    video_file = temp_dir / "test_video.mp4"
    video_file.touch()
    return str(video_file)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a mock audio file for testing"""
    audio_file = temp_dir / "test_audio.mp3"
    audio_file.touch()
    return str(audio_file)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a mock text file for testing (unsupported format)"""
    text_file = temp_dir / "test_file.txt"
    text_file.touch()
    return str(text_file)
