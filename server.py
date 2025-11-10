import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP
import json, os, requests
import datetime as dt
import io, base64
import uvicorn

mcp = FastMCP("Weather")

API = "https://api.tomorrow.io/v4/weather/forecast"
API_RT = "https://api.tomorrow.io/v4/weather/realtime"

def _get(url, params):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def _now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def _plus_hours_iso(h):
    return (dt.datetime.utcnow() + dt.timedelta(hours=h)).replace(microsecond=0).isoformat()+"Z"

@mcp.tool()
def weather_forecast(q: str,
                     hours: int = 12,
                     step: str = "1h",
                     fields: str = "temperature,temperatureApparent,humidity,precipitationProbability,weatherCode,windSpeed,windGust,uvIndex",
                     compact: bool = True) -> str:
    """
    Small-JSON weather forecast via Tomorrow.io.
    Args:
      q: "lat,lon" or place name (e.g., "21.03,105.85" or "Hanoi")
      hours: limit forecast window (default 12)
      step: "1h" or "1d"
      fields: comma list of fields to fetch
      compact: if True, return a slimmed structure (recommended)
    Returns:
      JSON string with ok/source/ta. daData is compact to avoid token limits.
    """
    TomorrowAPI_Key = os.getenv("TomorrowAPI_Key")
    if not TomorrowAPI_Key:
        return json.dumps({"ok": False, "error": "TomorrowAPI_Key not set"})

    hours = max(1, min(int(hours), 48))
    params = {
        "location": q,
        "apikey": TomorrowAPI_Key,
        "timesteps": step,
        "startTime": "now",
        "endTime": f"nowPlus{hours}h",
    }

    if fields:
        params["fields"] = [f.strip() for f in fields.split(",") if f.strip()]

    try:
        raw = _get(API, params)
    except requests.HTTPError as e:
        return json.dumps({"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"},
                          separators=(",", ":"))
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, separators=(",", ":"))

    if not compact:
        return json.dumps({"ok": True, "source": "tomorrow", "data": raw},
                          separators=(",", ":"))

    try:
        hourly = raw.get("timelines", {}).get("hourly", [])
        slim = []
        for p in hourly[:hours]:
            v = p.get("values", {})
            slim.append({
                "t": p.get("time"),
                "temp": v.get("temperature"),
                "feel": v.get("temperatureApparent"),
                "h": v.get("humidity"),
                "pop": v.get("precipitationProbability"),
                "wcode": v.get("weatherCode"),
                "wind": v.get("windSpeed"),
                "gust": v.get("windGust"),
                "uv": v.get("uvIndex"),
            })
        loc = raw.get("location", {})
        out = {
            "ok": True,
            "source": "tomorrow",
            "loc": {"name": loc.get("name"), "lat": loc.get("lat"), "lon": loc.get("lon")},
            "step": step,
            "range": {"start": _now_iso(), "end": _plus_hours_iso(hours)},
            "points": slim
        }
        return json.dumps(out, separators=(",", ":"))
    except Exception as e:
        return json.dumps({"ok": True, "source": "tomorrow", "data": raw, "warn": f"compact_failed:{e}"},
                          separators=(",", ":"))

@mcp.tool()
def weather_now(q: str,
                fields: str = "temperature,temperatureApparent,humidity,weatherCode,windSpeed,uvIndex") -> str:
    """
    Ultra-small realtime weather.
    """
    TomorrowAPI_Key = os.getenv("TomorrowAPI_Key")
    if not TomorrowAPI_Key:
        return json.dumps({"ok": False, "error": "TomorrowAPI_Key not set"})
    params = {"location": q, "apikey": TomorrowAPI_Key}
    if fields:
        params["fields"] = [f.strip() for f in fields.split(",") if f.strip()]
    try:
        data = _get(API_RT, params)
        v = data.get("data", {}).get("values", {})
        out = {
            "ok": True,
            "source": "tomorrow",
            "loc": data.get("location"),
            "obs": {
                "t": data.get("data", {}).get("time"),
                "temp": v.get("temperature"),
                "feel": v.get("temperatureApparent"),
                "h": v.get("humidity"),
                "wcode": v.get("weatherCode"),
                "wind": v.get("windSpeed"),
                "uv": v.get("uvIndex")
            }
        }
        return json.dumps(out, separators=(",", ":"))
    except requests.HTTPError as e:
        return json.dumps({"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"},
                          separators=(",", ":"))
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, separators=(",", ":"))

if __name__ == '__main__':
    port=int(os.getenv("PORT",8000))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
