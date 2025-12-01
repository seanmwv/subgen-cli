"""Tests for language code conversion"""

import pytest
from subgen_cli.cli import LanguageCode


class TestLanguageCodeFromISO6391:
    """Test conversion from ISO 639-1 codes"""

    def test_common_languages(self):
        """Test common language codes"""
        assert LanguageCode.from_iso_639_1('en') == LanguageCode.ENGLISH
        assert LanguageCode.from_iso_639_1('es') == LanguageCode.SPANISH
        assert LanguageCode.from_iso_639_1('fr') == LanguageCode.FRENCH
        assert LanguageCode.from_iso_639_1('de') == LanguageCode.GERMAN
        assert LanguageCode.from_iso_639_1('ja') == LanguageCode.JAPANESE
        assert LanguageCode.from_iso_639_1('zh') == LanguageCode.CHINESE

    def test_invalid_code(self):
        """Test invalid ISO 639-1 code returns NONE"""
        assert LanguageCode.from_iso_639_1('xx') == LanguageCode.NONE
        assert LanguageCode.from_iso_639_1('invalid') == LanguageCode.NONE

    def test_none_input(self):
        """Test None input returns NONE"""
        result = LanguageCode.from_iso_639_1(None)
        assert result == LanguageCode.NONE


class TestLanguageCodeFromISO6392:
    """Test conversion from ISO 639-2 codes"""

    def test_iso_639_2_t_codes(self):
        """Test ISO 639-2/T codes"""
        assert LanguageCode.from_iso_639_2('eng') == LanguageCode.ENGLISH
        assert LanguageCode.from_iso_639_2('spa') == LanguageCode.SPANISH
        assert LanguageCode.from_iso_639_2('fra') == LanguageCode.FRENCH

    def test_iso_639_2_b_codes(self):
        """Test ISO 639-2/B codes (bibliographic)"""
        assert LanguageCode.from_iso_639_2('ger') == LanguageCode.GERMAN
        assert LanguageCode.from_iso_639_2('deu') == LanguageCode.GERMAN

    def test_invalid_code(self):
        """Test invalid ISO 639-2 code returns NONE"""
        assert LanguageCode.from_iso_639_2('xxx') == LanguageCode.NONE


class TestLanguageCodeFromName:
    """Test conversion from language names"""

    def test_english_names(self):
        """Test conversion from English language names"""
        assert LanguageCode.from_name('English') == LanguageCode.ENGLISH
        assert LanguageCode.from_name('Spanish') == LanguageCode.SPANISH
        assert LanguageCode.from_name('French') == LanguageCode.FRENCH

    def test_native_names(self):
        """Test conversion from native language names"""
        assert LanguageCode.from_name('Español') == LanguageCode.SPANISH
        assert LanguageCode.from_name('Français') == LanguageCode.FRENCH
        assert LanguageCode.from_name('Deutsch') == LanguageCode.GERMAN
        assert LanguageCode.from_name('日本語') == LanguageCode.JAPANESE
        assert LanguageCode.from_name('中文') == LanguageCode.CHINESE

    def test_case_insensitive(self):
        """Test name matching is case-insensitive"""
        assert LanguageCode.from_name('english') == LanguageCode.ENGLISH
        assert LanguageCode.from_name('SPANISH') == LanguageCode.SPANISH
        assert LanguageCode.from_name('FrEnCh') == LanguageCode.FRENCH


class TestLanguageCodeFromString:
    """Test conversion from any string format"""

    def test_iso_639_1(self):
        """Test from_string accepts ISO 639-1"""
        assert LanguageCode.from_string('en') == LanguageCode.ENGLISH
        assert LanguageCode.from_string('es') == LanguageCode.SPANISH

    def test_iso_639_2(self):
        """Test from_string accepts ISO 639-2"""
        assert LanguageCode.from_string('eng') == LanguageCode.ENGLISH
        assert LanguageCode.from_string('spa') == LanguageCode.SPANISH

    def test_english_name(self):
        """Test from_string accepts English names"""
        assert LanguageCode.from_string('English') == LanguageCode.ENGLISH
        assert LanguageCode.from_string('Spanish') == LanguageCode.SPANISH

    def test_native_name(self):
        """Test from_string accepts native names"""
        assert LanguageCode.from_string('Español') == LanguageCode.SPANISH
        assert LanguageCode.from_string('Français') == LanguageCode.FRENCH

    def test_whitespace_handling(self):
        """Test from_string handles whitespace"""
        assert LanguageCode.from_string('  en  ') == LanguageCode.ENGLISH
        assert LanguageCode.from_string('  Spanish  ') == LanguageCode.SPANISH

    def test_none_input(self):
        """Test from_string with None returns NONE"""
        assert LanguageCode.from_string(None) == LanguageCode.NONE

    def test_invalid_string(self):
        """Test from_string with invalid string returns NONE"""
        assert LanguageCode.from_string('invalid') == LanguageCode.NONE
        assert LanguageCode.from_string('xyz') == LanguageCode.NONE


class TestLanguageCodeValidation:
    """Test language code validation"""

    def test_is_valid_language(self):
        """Test is_valid_language method"""
        assert LanguageCode.is_valid_language('en')
        assert LanguageCode.is_valid_language('es')
        assert LanguageCode.is_valid_language('eng')
        assert LanguageCode.is_valid_language('English')
        assert LanguageCode.is_valid_language('Español')

    def test_is_invalid_language(self):
        """Test is_valid_language returns False for invalid codes"""
        assert not LanguageCode.is_valid_language('invalid')
        assert not LanguageCode.is_valid_language('xyz')
        assert not LanguageCode.is_valid_language('')


class TestLanguageCodeConversion:
    """Test language code output conversions"""

    def test_to_iso_639_1(self):
        """Test conversion to ISO 639-1"""
        assert LanguageCode.ENGLISH.to_iso_639_1() == 'en'
        assert LanguageCode.SPANISH.to_iso_639_1() == 'es'
        assert LanguageCode.FRENCH.to_iso_639_1() == 'fr'

    def test_to_iso_639_2_t(self):
        """Test conversion to ISO 639-2/T"""
        assert LanguageCode.ENGLISH.to_iso_639_2_t() == 'eng'
        assert LanguageCode.SPANISH.to_iso_639_2_t() == 'spa'
        assert LanguageCode.FRENCH.to_iso_639_2_t() == 'fra'

    def test_to_iso_639_2_b(self):
        """Test conversion to ISO 639-2/B"""
        assert LanguageCode.ENGLISH.to_iso_639_2_b() == 'eng'
        assert LanguageCode.GERMAN.to_iso_639_2_b() == 'ger'

    def test_to_name_english(self):
        """Test conversion to English name"""
        assert LanguageCode.ENGLISH.to_name(in_english=True) == 'English'
        assert LanguageCode.SPANISH.to_name(in_english=True) == 'Spanish'
        assert LanguageCode.FRENCH.to_name(in_english=True) == 'French'

    def test_to_name_native(self):
        """Test conversion to native name"""
        assert LanguageCode.SPANISH.to_name(in_english=False) == 'Español'
        assert LanguageCode.FRENCH.to_name(in_english=False) == 'Français'
        assert LanguageCode.GERMAN.to_name(in_english=False) == 'Deutsch'


class TestLanguageCodeSpecialCases:
    """Test special language code cases"""

    def test_none_language(self):
        """Test NONE language code"""
        assert LanguageCode.NONE.to_iso_639_1() is None
        assert not bool(LanguageCode.NONE)

    def test_language_equality(self):
        """Test language code equality"""
        assert LanguageCode.ENGLISH == LanguageCode.ENGLISH
        assert LanguageCode.ENGLISH != LanguageCode.SPANISH
        assert LanguageCode.NONE == None

    def test_language_string_representation(self):
        """Test string representation of language codes"""
        assert str(LanguageCode.ENGLISH) == 'English'
        assert str(LanguageCode.SPANISH) == 'Spanish'
        assert str(LanguageCode.NONE) == 'Unknown'

    def test_language_boolean_conversion(self):
        """Test boolean conversion of language codes"""
        assert bool(LanguageCode.ENGLISH)
        assert bool(LanguageCode.SPANISH)
        assert not bool(LanguageCode.NONE)

    def test_cantonese_special_case(self):
        """Test Cantonese (special 3-letter ISO 639-1 code)"""
        assert LanguageCode.CANTONESE.iso_639_1 == 'yue'
        assert LanguageCode.from_string('yue') == LanguageCode.CANTONESE
        assert LanguageCode.from_string('Cantonese') == LanguageCode.CANTONESE
