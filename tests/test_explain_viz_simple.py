import xml.etree.ElementTree as ET

from datacreek.analysis.explain_viz import explain_to_svg


def test_explain_to_svg_empty():
    svg = explain_to_svg({"nodes": []})
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_explain_to_svg_graph():
    data = {
        "nodes": ["A", "B"],
        "edges": [("A", "B")],
        "attention": {"A->B": 0.5},
    }
    svg = explain_to_svg(data)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "<line" in svg  # one edge drawn
    assert "stroke=" in svg
