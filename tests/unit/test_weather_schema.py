import pytest

from pydantic import ValidationError

from citymind.ingestion.schemas import WeatherMeasurement


def valid_weather():
    return {
        "time": "2026-11-22T11:45",
        "temperature_2m": 20.4,
        "relative_humidity_2m": 43,
        "wind_speed_10m": 4.6,
    }


def test_valid_weather():
    weather = WeatherMeasurement.model_validate(
        valid_weather()
    )

    assert weather.temperature_2m == 20.4
    assert weather.relative_humidity_2m == 43
    assert weather.wind_speed_10m == 4.6


def test_invalid_humidity():
    data = valid_weather()
    data["relative_humidity_2m"] = 150

    with pytest.raises(ValidationError):
        WeatherMeasurement.model_validate(data)


def test_negative_wind_speed():
    data = valid_weather()
    data["wind_speed_10m"] = -5

    with pytest.raises(ValidationError):
        WeatherMeasurement.model_validate(data)


def test_missing_temperature():
    data = valid_weather()
    del data["temperature_2m"]

    with pytest.raises(ValidationError):
        WeatherMeasurement.model_validate(data)


def test_unexpected_field():
    data = valid_weather()
    data["unknown_sensor"] = 999

    with pytest.raises(ValidationError):
        WeatherMeasurement.model_validate(data)


def test_invalid_timestamp():
    data = valid_weather()
    data["time"] = "not-a-date"

    with pytest.raises(ValidationError):
        WeatherMeasurement.model_validate(data)