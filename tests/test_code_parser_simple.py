import pytest
from datacreek.parsers.code_parser import CodeParser


def test_code_parser_read(tmp_path):
    code_file = tmp_path / "sample.py"
    code_file.write_text("print('hi')\n", encoding="utf-8")
    parser = CodeParser()
    assert parser.parse(str(code_file)) == "print('hi')\n"


def test_code_parser_missing():
    parser = CodeParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("missing_file.py")

