from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template

###############################################################################
# Flask setup
###############################################################################
app = Flask(__name__)

# Common IATA airline codes → full names (extend as needed)
AIRLINE_NAMES = {
    # European majors & regionals
    "A3": "Aegean Airlines",
    "AF": "Air France",
    "AZ": "ITA Airways",
    "BA": "British Airways",
    "E9": "Iberojet",
    "IB": "Iberia",
    "KL": "KLM",
    "LH": "Lufthansa",
    "NT": "Binter Canarias",
    "PC": "Pegasus Airlines",
    "TP": "TAP Air Portugal",
    "VY": "Vueling",
    "UX": "Air Europa",


    # Americas
    "AA": "American Airlines",
    "AV": "Avianca",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "TS": "Air Transat",
    "UA": "United Airlines",
    "2W": "World2Fly",

    # Middle East & Africa
    "AT": "Royal Air Maroc",
    "EK": "Emirates",
    "MS": "EgyptAir",
    "TK": "Turkish Airlines",

    # Asia‑Pacific
    "CA": "Air China",
    "MU": "China Eastern Airlines",
    # Add more codes below as needed…
}

###############################################################################
# Helper functions (same logic you had in the CLI script)
###############################################################################
def load_offers(path: str | Path = "flights.json") -> list[dict]:
    """Return the list with flight‑offer objects, regardless of dump format."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("data", [])


def nice_time(iso_str: str) -> str:
    """'2025‑07‑26T09:20:00' ➜ '09:20'."""
    return datetime.fromisoformat(iso_str.rstrip("Z")).strftime("%H:%M")


###############################################################################
# A tiny custom Jinja filter so we can call |timenice in the template
###############################################################################
@app.template_filter("timenice")
def timenice_filter(value: str) -> str:      # noqa: D401
    return nice_time(value)


###############################################################################
# Single page
###############################################################################
@app.route("/", methods=["GET"])
def index():
    offers = load_offers()
    rows: list[dict] = []

    # ---------------------------------------------------------------
    # 1) Build plain rows, capturing seats information
    # ---------------------------------------------------------------
    for offer in offers:
        first_seg   = offer["itineraries"][0]["segments"][0]
        last_seg    = offer["itineraries"][0]["segments"][-1]

        carrier     = first_seg["carrierCode"]
        airline     = AIRLINE_NAMES.get(carrier, carrier)        # fallback to code
        flight_code = f"{carrier}{first_seg['number']}"
        dep_time_iso = first_seg["departure"]["at"]

        price_str   = offer["price"]["total"]
        price_val   = float(price_str)

        seats       = int(offer.get("numberOfBookableSeats", 0))

        rows.append(
            {
                "origin"     : first_seg["departure"]["iataCode"],
                "destination": last_seg["arrival"]["iataCode"],
                "airline"    : airline,
                "flight"     : flight_code,
                "price"      : price_str,
                "price_val"  : price_val,         # helper for sorting
                "currency"   : offer["price"]["currency"],
                "time_iso"   : dep_time_iso,
                "seats"      : seats,
            }
        )

    # ---------------------------------------------------------------
    # 2) Deduplicate:
    #    - same origin & destination
    #    - departure within 20 minutes
    #    - seats within ±2
    #    → keep cheapest (rows are pre‑sorted by price)
    # ---------------------------------------------------------------
    rows.sort(key=lambda r: r["price_val"])           # cheapest first
    unique_rows: list[dict] = []

    for row in rows:
        row_time = datetime.fromisoformat(row["time_iso"].rstrip("Z"))

        duplicate_found = False
        for kept in unique_rows:
            if (
                row["origin"] == kept["origin"]
                and row["destination"] == kept["destination"]
                and abs(
                    (row_time - datetime.fromisoformat(kept["time_iso"].rstrip("Z")))
                    .total_seconds()
                )
                <= 20 * 60
                and abs(row["seats"] - kept["seats"]) <= 2
            ):
                duplicate_found = True
                break

        if not duplicate_found:
            unique_rows.append(row)

    # ---------------------------------------------------------------
    # 3) Final ordering (chronological) and numbering column
    # ---------------------------------------------------------------
    rows = sorted(unique_rows, key=lambda r: r["time_iso"])

    width = len(str(len(rows)))
    for idx, row in enumerate(rows, start=1):
        row["num"] = f"{idx:0{width}d}"
        row.pop("price_val", None)

    return render_template("index.html", offers=rows)


###############################################################################
# Dev server
###############################################################################
if __name__ == "__main__":
    # debug=True ➜ auto‑reload on code changes
    app.run(debug=True, host="0.0.0.0", port=5000)