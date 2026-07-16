"""Unit tests for VLM response parsing — runs fast, no GPU, no API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from label import parse


def test_numeric_class_id():
    idx, conf = parse('{"class_id": 14, "class_name": "Stop", "confidence": 0.95}')
    assert idx == 14
    assert abs(conf - 0.95) < 1e-6


def test_class_id_takes_priority_over_name():
    # class_id 0 = Speed limit 20, but name says Stop
    idx, conf = parse('{"class_id": 0, "class_name": "Stop", "confidence": 0.8}')
    assert idx == 0


def test_name_fallback_when_id_invalid():
    idx, conf = parse('{"class_id": 999, "class_name": "Stop", "confidence": 0.7}')
    assert idx == 14


def test_fuzzy_name_match():
    idx, conf = parse('{"class_id": null, "class_name": "priority road sign", "confidence": 0.75}')
    assert idx == 12


def test_preamble_ignored():
    idx, conf = parse('Sure! {"class_id": 5, "class_name": "Speed limit 80", "confidence": 0.9} done')
    assert idx == 5
    assert abs(conf - 0.9) < 1e-6


def test_invalid_json_returns_none():
    idx, conf = parse("I cannot classify this image")
    assert idx is None
    assert conf == 0.0


def test_boolean_class_id_rejected():
    idx, _ = parse('{"class_id": true, "confidence": 0.9}')
    assert idx is None


def test_out_of_range_class_id_rejected():
    idx, _ = parse('{"class_id": 43, "confidence": 0.9}')
    assert idx is None


def test_zero_confidence():
    idx, conf = parse('{"class_id": 2, "class_name": "Speed limit 50", "confidence": 0.0}')
    assert idx == 2
    assert conf == 0.0
