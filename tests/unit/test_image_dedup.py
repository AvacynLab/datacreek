from PIL import Image

import datacreek.utils.image_dedup as dedup


def create_image(color, path):
    """Create a simple square image of a given color."""
    img = Image.new('RGB', (8, 8), color)
    img.save(path)


def test_hashes_count():
    h = list(dedup._hashes(b'data'))
    assert len(h) == dedup.K
    assert all(0 <= x < dedup.M for x in h)


def test_check_duplicate(tmp_path, monkeypatch):
    # reset filter to avoid cross-test interference
    monkeypatch.setattr(dedup, 'FILTER', bytearray(dedup._BYTES))
    p1 = tmp_path / 'a.png'
    create_image('red', p1)
    assert dedup.check_duplicate(str(p1)) is False
    # same image again -> duplicate
    assert dedup.check_duplicate(str(p1)) is True
    # different image should not be duplicate
    p2 = tmp_path / 'b.png'
    img = Image.new('RGB', (8, 8), 'blue')
    img.putpixel((0, 0), (255, 0, 0))
    img.save(p2)
    assert dedup.check_duplicate(str(p2)) is False
