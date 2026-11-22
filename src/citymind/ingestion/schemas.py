from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WeatherMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: datetime

    temperature_2m: float = Field(
        ge=-100,
        le=70,
    )

    relative_humidity_2m: float = Field(
        ge=0,
        le=100,
    )

    wind_speed_10m: float = Field(
        ge=0,
        le=500,
    )