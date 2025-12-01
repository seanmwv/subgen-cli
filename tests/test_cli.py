"""Tests for CLI argument parsing"""

import pytest
import sys
from subgen_cli.cli import parse_arguments


def test_parse_basic_arguments(monkeypatch):
    """Test parsing basic required arguments"""
    test_args = ['subgen', '-f', 'test.mp4']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.file == 'test.mp4'
    assert args.model == 'medium'  # default
    assert args.device == 'cpu'  # default
    assert args.task == 'transcribe'  # default
    assert args.model_path is None  # default (will use platform-specific path)


def test_parse_model_argument(monkeypatch):
    """Test parsing model selection"""
    test_args = ['subgen', '-f', 'test.mp4', '--model', 'small']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.model == 'small'


def test_parse_device_argument(monkeypatch):
    """Test parsing device selection"""
    test_args = ['subgen', '-f', 'test.mp4', '--device', 'cuda']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.device == 'cuda'


def test_parse_language_argument(monkeypatch):
    """Test parsing language argument"""
    test_args = ['subgen', '-f', 'test.mp4', '-l', 'es']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.language == 'es'


def test_parse_task_argument(monkeypatch):
    """Test parsing task argument"""
    test_args = ['subgen', '-f', 'test.mp4', '-t', 'translate']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.task == 'translate'


def test_parse_audio_track_argument(monkeypatch):
    """Test parsing audio track selection"""
    test_args = ['subgen', '-f', 'test.mp4', '-a', '2']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.audio_track == 2


def test_parse_output_directory(monkeypatch):
    """Test parsing output directory"""
    test_args = ['subgen', '-f', 'test.mp4', '-o', './subtitles']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.output == './subtitles'


def test_parse_model_path(monkeypatch):
    """Test parsing custom model path"""
    test_args = ['subgen', '-f', 'test.mp4', '--model-path', '/custom/models']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.model_path == '/custom/models'


def test_parse_word_highlight(monkeypatch):
    """Test parsing word highlight flag"""
    test_args = ['subgen', '-f', 'test.mp4', '-w']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.word_highlight is True


def test_parse_verbose_flag(monkeypatch):
    """Test parsing verbose flag"""
    test_args = ['subgen', '-f', 'test.mp4', '-v']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.verbose is True


def test_parse_no_lrc_flag(monkeypatch):
    """Test parsing no-lrc flag"""
    test_args = ['subgen', '-f', 'test.mp4', '--no-lrc']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.no_lrc is True


def test_parse_no_model_name_flag(monkeypatch):
    """Test parsing no-model-name flag"""
    test_args = ['subgen', '-f', 'test.mp4', '--no-model-name']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.no_model_name is True


def test_parse_list_tracks_flag(monkeypatch):
    """Test parsing list-tracks flag"""
    test_args = ['subgen', '-f', 'test.mp4', '--list-tracks']
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.list_tracks is True


def test_parse_multiple_options(monkeypatch):
    """Test parsing multiple options together"""
    test_args = [
        'subgen', '-f', 'test.mp4',
        '--model', 'large',
        '--device', 'cuda',
        '-l', 'fr',
        '-t', 'translate',
        '-w',
        '-v'
    ]
    monkeypatch.setattr(sys, 'argv', test_args)

    args = parse_arguments()
    assert args.file == 'test.mp4'
    assert args.model == 'large'
    assert args.device == 'cuda'
    assert args.language == 'fr'
    assert args.task == 'translate'
    assert args.word_highlight is True
    assert args.verbose is True


def test_missing_required_file_argument(monkeypatch):
    """Test that missing required file argument raises error"""
    test_args = ['subgen']
    monkeypatch.setattr(sys, 'argv', test_args)

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_model_choice(monkeypatch):
    """Test that invalid model choice raises error"""
    test_args = ['subgen', '-f', 'test.mp4', '--model', 'invalid']
    monkeypatch.setattr(sys, 'argv', test_args)

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_device_choice(monkeypatch):
    """Test that invalid device choice raises error"""
    test_args = ['subgen', '-f', 'test.mp4', '--device', 'invalid']
    monkeypatch.setattr(sys, 'argv', test_args)

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_task_choice(monkeypatch):
    """Test that invalid task choice raises error"""
    test_args = ['subgen', '-f', 'test.mp4', '-t', 'invalid']
    monkeypatch.setattr(sys, 'argv', test_args)

    with pytest.raises(SystemExit):
        parse_arguments()
