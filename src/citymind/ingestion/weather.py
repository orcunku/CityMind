import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from citymind.ingestion.schemas import WeatherMeasurement


API_URL = "https://api.open-meteo.com/v1/forecast"

IZMIR_LAT = 38.4237
IZMIR_LON = 27.1428


def fetch_izmir_weather():
    # 1. API parametrelerini hazırla
    params = {
        "latitude": IZMIR_LAT,
        "longitude": IZMIR_LON,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "timezone": "UTC",
    }

    # 2. Open-Meteo API'den veri çek
    with httpx.Client(timeout=20.0) as client:
        response = client.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

    # 3. API yanıtının yapısını kontrol et
    if not isinstance(data, dict):
        raise ValueError("Invalid API response")

    current = data.get("current")

    if not isinstance(current, dict):
        raise ValueError("Missing current weather data")

    # 4. Gerekli alanları kontrol et
    required_fields = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]

    for field in required_fields:
        if current.get(field) is None:
            raise ValueError(
                f"Missing weather field: {field}"
            )

    # 5. Pydantic ile veri doğrulama
    validated = WeatherMeasurement.model_validate(
        {
            "time": current["time"],
            "temperature_2m": current["temperature_2m"],
            "relative_humidity_2m": current["relative_humidity_2m"],
            "wind_speed_10m": current["wind_speed_10m"],
        }
    )

    print("Weather data validation successful!")

    # 6. Kaynak ve zaman bilgilerini ekle
    retrieved_at = datetime.now(timezone.utc)

    record = {
        "source": "Open-Meteo",
        "source_url": str(response.url),
        "data_type": "modeled_weather",
        "location": "Izmir, Turkiye",
        "retrieved_at": retrieved_at.isoformat(),
        "validated_data": validated.model_dump(
            mode="json"
        ),
        "data": data,
    }

    # 7. Kayıt klasörünü oluştur
    output_dir = Path("data/raw")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 8. Benzersiz dosya adı oluştur
    timestamp = retrieved_at.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_file = (
        output_dir / f"weather_{timestamp}.json"
    )

    # 9. JSON dosyasına kaydet
    output_file.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Weather data successfully fetched!")
    print(f"Saved to: {output_file}")

    print(
        json.dumps(
            current,
            indent=2,
            ensure_ascii=False,
        )
    )

    return record


if __name__ == "__main__":
    fetch_izmir_weather()
