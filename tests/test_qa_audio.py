from __future__ import annotations

import numpy as np

from workers.qa_audio import uncovered_source_regions


def tone(seconds: float, sample_rate: int = 1000, level: float = .1) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * 90 * time) * level).astype(np.float32)


def test_detects_sustained_source_speech_missing_from_dub() -> None:
    sample_rate = 1000
    source = np.zeros(5 * sample_rate, dtype=np.float32)
    dub = np.zeros_like(source)
    source[1000:3000] = tone(2, sample_rate)
    dub[1000:1500] = tone(.5, sample_rate)

    regions = uncovered_source_regions(source, dub, sample_rate)

    assert len(regions) == 1
    assert regions[0]["start"] >= 1.9
    assert regions[0]["duration"] >= .8


def test_allows_short_natural_gaps_and_complete_delivery() -> None:
    sample_rate = 1000
    source = np.zeros(4 * sample_rate, dtype=np.float32)
    dub = np.zeros_like(source)
    source[500:3000] = tone(2.5, sample_rate)
    dub[500:1700] = tone(1.2, sample_rate)
    dub[1820:3000] = tone(1.18, sample_rate)

    assert uncovered_source_regions(source, dub, sample_rate) == []


def test_supports_different_source_and_dub_sample_rates() -> None:
    source_rate = 1000
    dub_rate = 2000
    source = np.zeros(3 * source_rate, dtype=np.float32)
    dub = np.zeros(3 * dub_rate, dtype=np.float32)
    source[500:2500] = tone(2, source_rate)
    dub[1000:5000] = tone(2, dub_rate)

    assert uncovered_source_regions(source, dub, source_rate, dub_rate) == []
