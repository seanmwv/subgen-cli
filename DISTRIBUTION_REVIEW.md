# Package Distribution Review - subgen-cli

**Review Date:** 2025-11-30
**Reviewer:** Claude Code
**Status:** 🔴 NEEDS FIXES BEFORE DISTRIBUTION

---

## CRITICAL ISSUES

### 1. pyproject.toml Syntax Errors

**Location:** `pyproject.toml:18-22`

**Issues:**
- Missing commas in dependencies array (lines 19-20)
- Invalid PyTorch version `2.9.1` - this version does not exist
- Syntax errors will prevent package from building

**Fix Required:**
```toml
dependencies = [
    "ffmpeg_python==0.2.0",
    "stable_ts[fw]==2.19.1",
    "torch>=2.0.0",
]
```

---

### 2. Invalid Entry Point Configuration

**Location:** `pyproject.toml:25`

**Issue:**
```toml
[project.scripts]
subgen = "subgen_cli:main"
```

The entry point `subgen_cli:main` is incorrect. The actual module path is `subgen-cli.cli:main` (but this is also problematic - see issue #3).

**Fix Required:**
After renaming package directory (see #3), use:
```toml
[project.scripts]
subgen = "subgen_cli.cli:main"
```

---

### 3. Invalid Package Directory Name

**Location:** `src/subgen-cli/`

**Issue:**
- Package directory uses hyphen: `subgen-cli`
- Python packages MUST use underscores or no separator
- Hyphens are not valid in Python identifiers
- This will cause import errors

**Fix Required:**
1. Rename `src/subgen-cli/` to `src/subgen_cli/`
2. Update `pyproject.toml` line 37:
   ```toml
   [tool.hatch.build]
   packages = ["src/subgen_cli"]
   ```

---

### 4. Missing `__init__.py`

**Location:** `src/subgen-cli/` (soon to be `src/subgen_cli/`)

**Issue:**
- No `__init__.py` file in package directory
- Package will not be recognized as a Python package
- Entry point will fail to import

**Fix Required:**
Create `src/subgen_cli/__init__.py` with:
```python
"""Subgen CLI - Standalone command-line interface for subtitle generation"""

__version__ = "1.0.0"

from .cli import main

__all__ = ["main", "__version__"]
```

Also update `cli.py` to import version from package:
```python
from . import __version__
```

---

## HIGH PRIORITY ISSUES

### 5. Inadequate .gitignore

**Location:** `.gitignore`

**Issues:**
- Typo: `.vnv` should be `.venv`
- Missing standard Python ignores

**Fix Required:**
Add to `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
.vnv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
models/
*.srt
*.lrc
```

---

### 6. Incomplete README.md

**Location:** `README.md:1-19`

**Issues:**
- Contains outdated installation instructions (`pip install -r requirements.txt`)
- References old usage pattern (`python cli.py`)
- No package installation instructions
- Minimal documentation
- Contains a "known issues" note that should be addressed or removed

**Fix Required:**
Rewrite README.md to include:
- Project description
- Features list
- Installation via pip
- Usage examples with installed command
- Configuration options
- Requirements (ffmpeg, etc.)
- Contributing guidelines
- License information
- Proper attribution to original McCloudS/subgen project

---

### 7. Redundant requirements.txt

**Location:** `requirements.txt`

**Issue:**
- Contains duplicate dependency specifications
- pyproject.toml is the source of truth for modern Python packages
- Can cause version conflicts

**Fix Required:**
Either:
1. Delete `requirements.txt` (recommended for pure package distribution)
2. OR generate it from pyproject.toml for development: `pip freeze > requirements.txt`
3. OR keep minimal requirements.txt that just references pyproject.toml:
   ```
   # Install the package in development mode
   -e .
   ```

---

### 8. License Attribution Issues

**Location:** `LICENSE` and `pyproject.toml`

**Issues:**
- LICENSE file: Copyright (c) 2023 McCloudS
- pyproject.toml: author is seanmwv
- Unclear licensing relationship

**Fix Required:**
1. Clarify if this is a fork or derivative work
2. If derivative, keep McCloudS copyright and add your own:
   ```
   Original work Copyright (c) 2023 McCloudS
   Modified work Copyright (c) 2025 seanmwv
   ```
3. Update pyproject.toml to reflect this
4. Add attribution in README.md acknowledging the original project

---

## MEDIUM PRIORITY ISSUES

### 9. Missing Package Metadata

**Location:** `pyproject.toml`

**Issues:**
- No `keywords` field for PyPI discoverability
- Limited classifiers
- No documentation URL
- No issue tracker URL

**Recommended Additions:**
```toml
keywords = ["subtitle", "whisper", "transcription", "ai", "video", "audio", "srt", "lrc"]

[project.urls]
Homepage = "https://github.com/seanmwv/subgen-cli"
Repository = "https://github.com/seanmwv/subgen-cli"
Documentation = "https://github.com/seanmwv/subgen-cli#readme"
"Bug Tracker" = "https://github.com/seanmwv/subgen-cli/issues"

# Add more classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Video",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

---

### 10. Version Synchronization

**Location:** `cli.py:12` and `pyproject.toml:3`

**Issue:**
- Version defined in two places: `cli.py` (__version__ = "1.0.0") and `pyproject.toml` (version = "0.0.0")
- Can get out of sync

**Fix Required:**
1. Update `pyproject.toml` version to "1.0.0" OR "0.1.0" for initial release
2. Use dynamic versioning:
   ```toml
   [project]
   dynamic = ["version"]

   [tool.hatch.version]
   path = "src/subgen_cli/__init__.py"
   ```

---

### 11. Missing Development Dependencies

**Location:** `pyproject.toml`

**Issue:**
- No development/testing dependencies specified
- Makes contribution difficult

**Recommended Addition:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
gpu = [
    "torch[cuda]>=2.0.0",
]
```

---

### 12. No Tests Directory

**Location:** Missing

**Issue:**
- No test suite included
- Cannot verify package works after installation

**Recommended:**
Create `tests/` directory with basic tests:
- Test CLI argument parsing
- Test file detection (audio/video)
- Test language code conversion
- Mock transcription tests

---

## LOW PRIORITY ISSUES

### 13. Missing MANIFEST.in

**Issue:**
- May be needed to include non-Python files (if any)
- Not critical if using hatchling

**Recommendation:**
Only needed if you want to include additional files like:
- Example files
- Documentation
- Configuration templates

---

### 14. No CHANGELOG.md

**Issue:**
- No version history tracking
- Difficult for users to see what changed between versions

**Recommendation:**
Create `CHANGELOG.md` following Keep a Changelog format

---

### 15. Example Directory in Package

**Location:** `example/marguerite duras on bresson.mkv`

**Issue:**
- 150+ MB video file in repository
- Not excluded from package distribution
- Will bloat package size

**Fix Required:**
Add to `.gitignore` and create `.gitattributes`:
```
example/*.mkv filter=lfs diff=lfs merge=lfs -text
```

OR exclude from package in pyproject.toml:
```toml
[tool.hatch.build.targets.wheel]
exclude = [
    "example/",
]
```

---

## DEPENDENCIES REVIEW

### Dependency Issues:

1. **torch==2.9.1** - Invalid version (doesn't exist)
   - Latest stable: 2.x series
   - Recommendation: `torch>=2.0.0,<3.0.0`

2. **ffmpeg_python==0.2.0** - Very specific pinning
   - Consider: `ffmpeg-python>=0.2.0`

3. **stable_ts[fw]==2.19.1** - Specific pinning
   - May cause conflicts
   - Consider: `stable-ts[fw]>=2.19.0,<3.0.0`

4. **Missing system dependencies:**
   - Package requires ffmpeg to be installed on system
   - Not documented in README or installation instructions

---

## PRE-DISTRIBUTION CHECKLIST

Before publishing to PyPI:

- [ ] Fix all CRITICAL issues (1-4)
- [ ] Fix all HIGH PRIORITY issues (5-8)
- [ ] Test package build: `python -m build`
- [ ] Test package installation: `pip install dist/subgen_cli-*.whl`
- [ ] Verify entry point works: `subgen --help`
- [ ] Test on clean virtual environment
- [ ] Update README with installation from PyPI
- [ ] Create git tag for version
- [ ] Test upload to TestPyPI first
- [ ] Verify package page on TestPyPI
- [ ] Only then upload to PyPI

---

## BUILD AND TEST COMMANDS

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Test install locally
pip install dist/subgen_cli-*.whl

# Test entry point
subgen --help

# Upload to TestPyPI (test first!)
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ subgen-cli

# Upload to PyPI (only after testing)
twine upload dist/*
```

---

## SUMMARY

**Total Issues Found:** 15
- Critical: 4
- High Priority: 4
- Medium Priority: 5
- Low Priority: 2

**Estimated Fix Time:** 2-4 hours

**Recommended Next Steps:**
1. Fix critical issues #1-4 (package structure and syntax)
2. Address high priority issues #5-8 (documentation and licensing)
3. Test package build and installation
4. Address medium priority issues for better PyPI presence
5. Consider low priority issues for long-term maintenance

---

## NOTES

- This package appears to be a CLI-focused fork of McCloudS/subgen
- Consider documenting relationship to upstream project
- GPU support should be optional dependency
- Consider adding examples and better documentation
- Package has good code quality, issues are mainly packaging-related
