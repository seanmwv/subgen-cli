"""Tests for transcription functionality (with mocking)"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from subgen_cli.cli import (
    TranscriptionConfig,
    WhisperModelManager,
    build_subtitle_filename,
    LanguageCode,
    get_default_model_path
)


class TestDefaultModelPath:
    """Test platform-specific default model path"""

    def test_default_model_path_windows(self):
        """Test default model path on Windows"""
        if os.name == 'nt':
            path = get_default_model_path()
            assert 'subgen' in path
            assert 'models' in path
            # Should use LOCALAPPDATA on Windows
            localappdata = os.getenv('LOCALAPPDATA')
            if localappdata:
                assert localappdata in path

    def test_default_model_path_unix(self):
        """Test default model path on Linux/macOS"""
        if os.name != 'nt':
            path = get_default_model_path()
            assert 'subgen' in path
            assert 'models' in path
            # Should use .cache on Unix systems
            assert '.cache' in path or os.path.expanduser('~') in path

    def test_default_model_path_not_relative(self):
        """Test that default path is absolute, not relative"""
        path = get_default_model_path()
        assert not path.startswith('./')
        assert not path.startswith('../')


class TestTranscriptionConfig:
    """Test TranscriptionConfig initialization"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TranscriptionConfig()

        assert config.model_name == 'medium'
        assert config.device == 'cpu'
        assert config.compute_type == 'auto'
        assert config.threads == 4
        assert config.model_path == get_default_model_path()
        assert config.task == 'transcribe'
        assert config.language is None
        assert config.word_highlight is False
        assert config.lrc_for_audio is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = TranscriptionConfig(
            model_name='large',
            device='cuda',
            language=LanguageCode.SPANISH,
            word_highlight=True,
            verbose=True
        )

        assert config.model_name == 'large'
        assert config.device == 'cuda'
        assert config.language == LanguageCode.SPANISH
        assert config.word_highlight is True
        assert config.verbose is True

    def test_custom_model_path(self):
        """Test custom model path overrides default"""
        custom_path = '/custom/path/to/models'
        config = TranscriptionConfig(model_path=custom_path)
        assert config.model_path == custom_path


class TestWhisperModelManager:
    """Test WhisperModelManager functionality"""

    def test_initialization(self):
        """Test model manager initialization"""
        manager = WhisperModelManager(
            model_name='small',
            device='cpu',
            compute_type='int8',
            threads=2,
            model_path='./test_models'
        )

        assert manager.model_name == 'small'
        assert manager.device == 'cpu'
        assert manager.compute_type == 'int8'
        assert manager.threads == 2
        assert manager.model_path == './test_models'

    def test_device_normalization(self):
        """Test device name normalization"""
        manager_cuda = WhisperModelManager(device='cuda', model_path='./test_models')
        manager_gpu = WhisperModelManager(device='gpu', model_path='./test_models')
        manager_cpu = WhisperModelManager(device='cpu', model_path='./test_models')

        assert manager_cuda.device == 'cuda'
        assert manager_gpu.device == 'cuda'
        assert manager_cpu.device == 'cpu'

    def test_default_model_path(self):
        """Test default model path is platform-specific"""
        manager = WhisperModelManager()
        assert manager.model_path == get_default_model_path()

    def test_is_loaded_initially_false(self):
        """Test model is not loaded initially"""
        manager = WhisperModelManager(model_path='./test_models')
        assert not manager.is_loaded()

    @patch('subgen_cli.cli.stable_whisper.load_faster_whisper')
    def test_load_model(self, mock_load):
        """Test model loading"""
        mock_model = Mock()
        mock_load.return_value = mock_model

        manager = WhisperModelManager(model_name='tiny', model_path='./test_models')
        manager.load()

        assert manager.is_loaded()
        mock_load.assert_called_once()

    @patch('subgen_cli.cli.stable_whisper.load_faster_whisper')
    def test_unload_model(self, mock_load):
        """Test model unloading"""
        mock_model = Mock()
        mock_model.model.unload_model = Mock()
        mock_load.return_value = mock_model

        manager = WhisperModelManager(model_path='./test_models')
        manager.load()
        assert manager.is_loaded()

        manager.unload()
        assert not manager.is_loaded()
        mock_model.model.unload_model.assert_called_once()


class TestSubtitleFilenameBuilder:
    """Test subtitle filename generation"""

    def test_basic_srt_filename(self):
        """Test basic SRT filename generation"""
        filename = build_subtitle_filename(
            base_path='/path/to/video',
            language=LanguageCode.ENGLISH,
            model_name='medium',
            is_lrc=False
        )

        assert filename == '/path/to/video.subgen.medium.eng.srt'

    def test_basic_lrc_filename(self):
        """Test basic LRC filename generation"""
        filename = build_subtitle_filename(
            base_path='/path/to/audio',
            language=LanguageCode.ENGLISH,
            model_name='small',
            is_lrc=True
        )

        assert filename == '/path/to/audio.subgen.small.lrc'

    def test_filename_without_subgen_tag(self):
        """Test filename without subgen tag"""
        filename = build_subtitle_filename(
            base_path='/path/to/video',
            language=LanguageCode.ENGLISH,
            model_name='medium',
            is_lrc=False,
            include_subgen=False
        )

        assert filename == '/path/to/video.medium.eng.srt'

    def test_filename_without_model_name(self):
        """Test filename without model name"""
        filename = build_subtitle_filename(
            base_path='/path/to/video',
            language=LanguageCode.ENGLISH,
            model_name='medium',
            is_lrc=False,
            include_model=False
        )

        assert filename == '/path/to/video.subgen.eng.srt'

    def test_filename_with_override(self):
        """Test filename with language override"""
        filename = build_subtitle_filename(
            base_path='/path/to/video',
            language=LanguageCode.ENGLISH,
            model_name='medium',
            is_lrc=False,
            subtitle_name_override='custom'
        )

        assert filename == '/path/to/video.subgen.medium.custom.srt'

    def test_filename_minimal(self):
        """Test minimal filename (no tags, no model)"""
        filename = build_subtitle_filename(
            base_path='/path/to/video',
            language=LanguageCode.SPANISH,
            model_name='large',
            is_lrc=False,
            include_model=False,
            include_subgen=False
        )

        assert filename == '/path/to/video.spa.srt'

    def test_filename_different_languages(self):
        """Test filename generation with different languages"""
        languages_and_codes = [
            (LanguageCode.SPANISH, 'spa'),
            (LanguageCode.FRENCH, 'fre'),
            (LanguageCode.GERMAN, 'ger'),
            (LanguageCode.JAPANESE, 'jpn'),
        ]

        for lang, expected_code in languages_and_codes:
            filename = build_subtitle_filename(
                base_path='/path/to/video',
                language=lang,
                model_name='medium',
                is_lrc=False
            )
            assert expected_code in filename


class TestMockTranscription:
    """Test transcription with mocked Whisper model"""

    @patch('subgen_cli.cli.whisper_model_context')
    @patch('subgen_cli.cli.os.path.exists')
    def test_transcribe_video_file(self, mock_exists, mock_context):
        """Test transcription of video file with mock"""
        from subgen_cli.cli import transcribe_file

        # Setup mocks
        mock_exists.return_value = True
        mock_manager = Mock()
        mock_model = Mock()
        mock_result = Mock()
        mock_result.language = 'en'
        mock_result.segments = []
        mock_result.to_srt_vtt = Mock()

        mock_model.transcribe.return_value = mock_result
        mock_manager.model = mock_model
        mock_context.return_value.__enter__ = Mock(return_value=mock_manager)
        mock_context.return_value.__exit__ = Mock(return_value=False)

        # Create config with explicit test path
        config = TranscriptionConfig(model_name='tiny', model_path='./test_models')

        # Transcribe
        output = transcribe_file('/path/to/video.mp4', config)

        # Verify
        assert output.endswith('.srt')
        mock_model.transcribe.assert_called_once()

    @patch('subgen_cli.cli.whisper_model_context')
    @patch('subgen_cli.cli.os.path.exists')
    def test_transcribe_with_language(self, mock_exists, mock_context):
        """Test transcription with forced language"""
        from subgen_cli.cli import transcribe_file

        # Setup mocks
        mock_exists.return_value = True
        mock_manager = Mock()
        mock_model = Mock()
        mock_result = Mock()
        mock_result.language = 'es'
        mock_result.segments = []
        mock_result.to_srt_vtt = Mock()

        mock_model.transcribe.return_value = mock_result
        mock_manager.model = mock_model
        mock_context.return_value.__enter__ = Mock(return_value=mock_manager)
        mock_context.return_value.__exit__ = Mock(return_value=False)

        # Create config with language and explicit test path
        config = TranscriptionConfig(
            model_name='tiny',
            language=LanguageCode.SPANISH,
            model_path='./test_models'
        )

        # Transcribe
        transcribe_file('/path/to/video.mp4', config)

        # Verify language was passed
        call_args = mock_model.transcribe.call_args
        assert call_args[1]['language'] == 'es'

    @patch('subgen_cli.cli.os.path.exists')
    def test_transcribe_nonexistent_file(self, mock_exists):
        """Test transcription of non-existent file raises error"""
        from subgen_cli.cli import transcribe_file

        mock_exists.return_value = False
        config = TranscriptionConfig(model_path='./test_models')

        with pytest.raises(FileNotFoundError):
            transcribe_file('/path/to/nonexistent.mp4', config)

    @patch('subgen_cli.cli.os.path.exists')
    def test_transcribe_unsupported_file(self, mock_exists):
        """Test transcription of unsupported file type raises error"""
        from subgen_cli.cli import transcribe_file

        mock_exists.return_value = True
        config = TranscriptionConfig(model_path='./test_models')

        with pytest.raises(ValueError):
            transcribe_file('/path/to/file.txt', config)
