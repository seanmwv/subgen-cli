"""Tests for audio/video file detection"""

import pytest
from subgen_cli.cli import is_audio_file, is_video_file, is_media_file


class TestVideoFileDetection:
    """Test video file extension detection"""

    def test_common_video_extensions(self):
        """Test common video file extensions"""
        video_files = [
            'video.mp4',
            'movie.mkv',
            'clip.avi',
            'film.mov',
            'video.wmv',
            'stream.webm',
        ]

        for video_file in video_files:
            assert is_video_file(video_file), f"{video_file} should be detected as video"
            assert is_media_file(video_file), f"{video_file} should be detected as media"

    def test_all_video_extensions(self):
        """Test all supported video extensions"""
        video_extensions = [
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mpg',
            '.mpeg', '.3gp', '.ogv', '.vob', '.rm', '.rmvb', '.ts', '.m4v',
            '.f4v', '.svq3', '.asf', '.m2ts', '.divx', '.xvid'
        ]

        for ext in video_extensions:
            test_file = f'test{ext}'
            assert is_video_file(test_file), f"{ext} should be detected as video"

    def test_case_insensitive_video(self):
        """Test that video detection is case-insensitive"""
        assert is_video_file('VIDEO.MP4')
        assert is_video_file('Video.MKV')
        assert is_video_file('video.Mp4')


class TestAudioFileDetection:
    """Test audio file extension detection"""

    def test_common_audio_extensions(self):
        """Test common audio file extensions"""
        audio_files = [
            'song.mp3',
            'audio.wav',
            'music.flac',
            'track.m4a',
            'podcast.aac',
        ]

        for audio_file in audio_files:
            assert is_audio_file(audio_file), f"{audio_file} should be detected as audio"
            assert is_media_file(audio_file), f"{audio_file} should be detected as media"

    def test_all_audio_extensions(self):
        """Test all supported audio extensions"""
        audio_extensions = [
            '.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.alac', '.m4a',
            '.opus', '.aiff', '.aif', '.pcm', '.ra', '.ram', '.mid', '.midi',
            '.ape', '.wv', '.amr', '.vox', '.tak', '.spx', '.m4b', '.mka'
        ]

        for ext in audio_extensions:
            test_file = f'test{ext}'
            assert is_audio_file(test_file), f"{ext} should be detected as audio"

    def test_case_insensitive_audio(self):
        """Test that audio detection is case-insensitive"""
        assert is_audio_file('SONG.MP3')
        assert is_audio_file('Song.Wav')
        assert is_audio_file('audio.Mp3')


class TestMediaFileDetection:
    """Test media file detection (audio or video)"""

    def test_media_file_includes_both(self):
        """Test that media file detection includes both audio and video"""
        assert is_media_file('video.mp4')
        assert is_media_file('audio.mp3')

    def test_non_media_files(self):
        """Test that non-media files are not detected"""
        non_media_files = [
            'document.txt',
            'image.jpg',
            'archive.zip',
            'script.py',
            'data.json',
            'subtitle.srt',
        ]

        for non_media in non_media_files:
            assert not is_audio_file(non_media), f"{non_media} should not be audio"
            assert not is_video_file(non_media), f"{non_media} should not be video"
            assert not is_media_file(non_media), f"{non_media} should not be media"

    def test_files_with_path(self):
        """Test file detection works with full paths"""
        assert is_video_file('/path/to/video.mp4')
        assert is_audio_file('/path/to/audio.mp3')
        assert is_video_file('C:\\Users\\videos\\movie.mkv')
        assert is_audio_file('C:\\Users\\music\\song.flac')

    def test_files_without_extension(self):
        """Test files without extensions are not detected"""
        assert not is_media_file('video')
        assert not is_media_file('audio')
        assert not is_media_file('/path/to/file')

    def test_empty_string(self):
        """Test empty string is not detected as media"""
        assert not is_media_file('')

    def test_dot_only(self):
        """Test dot-only filename is not detected as media"""
        assert not is_media_file('.')
        assert not is_media_file('..')
