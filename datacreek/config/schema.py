from __future__ import annotations

"""Typed configuration models."""

from pydantic import BaseModel, Field, ConfigDict


class PidConfig(BaseModel):
    """PID controller settings."""

    model_config = ConfigDict(extra="ignore")

    Kp: float = Field(0.4, gt=0.0, le=1.0, description="Proportional gain")
    Ki: float = Field(0.05, description="Integral gain")


class GpuConfig(BaseModel):
    """GPU acceleration options."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(False, description="Enable GPU features")


class ConfigSchema(BaseModel):
    """Root configuration schema."""

    model_config = ConfigDict(extra="allow")

    pid: PidConfig = Field(default_factory=PidConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
