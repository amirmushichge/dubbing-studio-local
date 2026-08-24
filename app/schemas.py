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
    lip_sync_enabled: bool = False
    subtitle_enabled: bool = True
    subtitle_style: str = "clean"
    subtitle_size: Literal["small", "medium", "large"] = "medium"
    subtitle_color: Literal["white", "yellow", "black"] = "white"
    subtitle_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    subtitle_x: float = Field(default=50.0, ge=5.0, le=95.0)
    subtitle_y: float = Field(default=88.0, ge=10.0, le=95.0)
    background_volume: float = Field(default=1.0, ge=0.0, le=1.0)
    expression: float = Field(default=0.5, ge=0.0, le=1.0)
    quality: Literal["draft", "balanced", "high"] = "high"
    burn_subtitles: bool = True


class CaptionRequest(BaseModel):
    subtitle_enabled: bool = True
    subtitle_style: str = "clean"
    subtitle_size: Literal["small", "medium", "large"] = "medium"
    subtitle_color: Literal["white", "yellow", "black"] = "white"
    subtitle_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    subtitle_x: float = Field(default=50.0, ge=5.0, le=95.0)
    subtitle_y: float = Field(default=88.0, ge=10.0, le=95.0)


class PreviewRequest(BaseModel):
    language: str
    voice_id: str
