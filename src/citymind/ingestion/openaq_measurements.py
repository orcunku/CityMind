import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


API_URL = "https://api.openaq.org/v3"

STATION_ID = 2946
SENSOR_ID = 2078778
POLLUTANT = "pm25"


def fetch_pm25_measurements():
    # API anahtarını yükle
    load_dotenv(
        dotenv_path=Path("/workspaces/CityMind/.env")
    )

    api_key = os.getenv("OPENAQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY is missing"
        )

    url = f"{API_URL}/sensors/{SENSOR_ID}/measurements"

    headers = {
        "X-API-Key": api_key,
    }

    params = {
        "limit": 10,
        "page": 1,
    }

    print("Fetching historical PM2.5 measurements...")

    # OpenAQ API isteği
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            url,
            headers=headers,
            params=params,
        )

        response.raise_for_status()
        data = response.json()

    # Yanıt kontrolü
    if not isinstance(data, dict):
        raise ValueError("Invalid API response")

    measurements = data.get("results")

    if not isinstance(measurements, list):
        raise ValueError(
            "Missing measurements list"
        )

    print(f"Measurements returned: {len(measurements)}")

    for measurement in measurements:
        print(
            "Value:",
            measurement.get("value"),
            "| Period:",
            measurement.get("period"),
        )

    # Kaynak bilgilerini kaydet
    retrieved_at = datetime.now(timezone.utc)

    record = {
        "source": "OpenAQ",
        "data_type": "historical_station_measurements",
        "station_id": STATION_ID,
        "sensor_id": SENSOR_ID,
        "pollutant": POLLUTANT,
        "unit": "µg/m³",
        "retrieved_at": retrieved_at.isoformat(),
        "source_url": str(response.url),
        "data": data,
    }

    output_dir = Path("data/raw")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = retrieved_at.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_file = (
        output_dir / f"openaq_pm25_{timestamp}.json"
    )

    output_file.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to: {output_file}")

    return record


if __name__ == "__main__":
    fetch_pm25_measurements()