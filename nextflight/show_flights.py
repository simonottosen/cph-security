"""
Pretty‑print every offer in flights.json as:

  01. ORIGIN : DESTINATION - PRICE CURRENCY - HH:MM

The time shown is the *scheduled departure* of the first segment.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def load_offers(path: str | Path = "flights.json") -> list[dict]:
    """
    Read the JSON file and return the list with flight‑offer objects.

    The Amadeus SDK stores the response body exactly in `Response.data`,
    so if you did `json.dump(response.data, ...)` then the root of the
    file is a *list*.  If instead you dumped the whole Response (e.g.
    `json.dump(response.result, ...)`) the list is under the "data" key.
    This helper accepts either form.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Detect where the offers list lives
    return raw if isinstance(raw, list) else raw.get("data", [])


def nice_time(iso_str: str) -> str:
    """
    Convert an ISO 8601 timestamp such as
    '2025-07-26T09:20:00'  ➜  '09:20'

    The Amadeus sandbox *does not* include time‑zone offsets,
    so `fromisoformat` is enough.  If you are in production and
    receive a trailing 'Z' or '+hh:mm', strip it or use
    dateutil.parser.isoparse for full flexibility.
    """
    return datetime.fromisoformat(iso_str.rstrip("Z")).strftime("%H:%M")


def print_offers(offers: list[dict]) -> None:
    """
    Iterate through the offers and print them in a tidy table‑like format.
    """
    width = len(str(len(offers)))          # 02‑d if <100 offers, 003‑d if 100+, …
    for idx, offer in enumerate(offers, start=1):
        # first itinerary ▸ first segment = departure
        first_seg = offer["itineraries"][0]["segments"][0]
        last_seg  = offer["itineraries"][0]["segments"][-1]

        origin      = first_seg["departure"]["iataCode"]
        destination = last_seg["arrival"]["iataCode"]
        price       = offer["price"]["total"]
        currency    = offer["price"]["currency"]
        dep_time    = nice_time(first_seg["departure"]["at"])

        print(f"{idx:0{width}d}. {origin} : {destination} - {price} {currency} - {dep_time}")


if __name__ == "__main__":
    offers = load_offers("flights.json")
    if not offers:
        raise SystemExit("❌  No offers found in flights.json")

    print_offers(offers)