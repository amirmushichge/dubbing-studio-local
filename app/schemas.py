from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    source_language: str = "auto"
    speaker_count: int | Literal["auto"] = "auto"


class SegmentPatch(BaseModel):
    segments: list[dict[str, Any]]
    speakers: list[dict[str, Any]] | None = None


class RenderRequest(BaseModel):
    target_language: str
    voice_mode: Literal["clone", "catalog"] = "clone"
    voice_id: str | None = None
    subtitle_enabled: bool = True
    subtitle_style: str = "clean"
    background_volume: float = Field(default=0.42, ge=0.0, le=1.0)
    expression: float = Field(default=0.5, ge=0.0, le=1.0)
    quality: Literal["draft", "balanced", "high"] = "high"
    burn_subtitles: bool = True


class PreviewRequest(BaseModel):
    language: str
    voice_id: str

