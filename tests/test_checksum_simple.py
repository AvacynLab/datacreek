from pathlib import Path
import pytest

from datacreek.utils.checksum import md5_file


def test_md5_file_valid(tmp_path):
    # Create a file with known contents and verify its MD5 checksum.
    p = tmp_path / "file.txt"
    p.write_text("hello")
    assert md5_file(str(p)) == "5d41402abc4b2a76b9719d911017c592"


def test_md5_file_missing(tmp_path):
    # When the target file does not exist a FileNotFoundError should surface.
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        md5_file(str(missing))
