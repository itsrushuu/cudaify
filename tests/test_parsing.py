"""Tests for parsing utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.nemotron import _parse_sections, strip_code_fences


def test_parse_sections_basic():
    content = (
        "CUDA KERNEL:\n```cpp\n__global__ void k() {}\n```\n"
        "HOST CODE:\n```cpp\nvoid launch() {}\n```\n"
        "EXPLANATION:\nA simple kernel.\n"
    )
    sections = _parse_sections(content)
    assert "CUDA KERNEL" in sections
    assert "__global__ void k()" in sections["CUDA KERNEL"]
    assert "HOST CODE" in sections
    assert "void launch()" in sections["HOST CODE"]
    assert "EXPLANATION" in sections
    assert sections["EXPLANATION"] == "A simple kernel."


def test_strip_code_fences():
    text = "```cpp\n__global__ void k() {}\n```"
    result = strip_code_fences(text)
    assert "```" not in result
    assert "__global__ void k()" in result


def test_parse_sections_no_headers():
    content = "Just some random text with no recognized headers."
    sections = _parse_sections(content)
    assert "preamble" in sections
    assert sections["preamble"] == content.strip()