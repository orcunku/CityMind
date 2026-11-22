import os

import httpx
from dotenv import load_dotenv


API_URL = "https://api.openaq.org/v3/locations"

IZMIR_LAT = 38.4237
IZMIR_LON = 27.1428


def discover_izmir_stations():
    load_dotenv()

    api_key = os.getenv("OPENAQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY is missing from .env"
        )

    headers = {
        "X-API-Key": api_key,
    }

    params = {
        "coordinates": f"{IZMIR_LAT},{IZMIR_LON}",
        "radius": 25000,
        "limit": 100,
        "page": 1,
    }

    print("Searching for air quality stations near Izmir...")

    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            API_URL,
            headers=headers,
            params=params,
        )

        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Invalid OpenAQ response")

    stations = data.get("results")

    if not isinstance(stations, list):
        raise ValueError("Missing station results")

    print(f"Stations returned: {len(stations)}")

    for station in stations:
        station_id = station.get("id")
        name = station.get("name")
        last = station.get("datetimeLast")

        print(f"\nStation ID: {station_id}")
        print(f"Name: {name}")
        print(f"Last data: {last}")

        sensors = station.get("sensors") or []

        for sensor in sensors:
            parameter = sensor.get("parameter") or {}

            print(
                "  Pollutant:",
                parameter.get("name"),
                "| Unit:",
                parameter.get("units"),
            )

    if not stations:
        print(
            "No stations returned within 25 km. "
            "This does not prove that Izmir has "
            "no air quality monitoring stations."
        )

    return stations


if __name__ == "__main__":
    discover_izmir_stations()