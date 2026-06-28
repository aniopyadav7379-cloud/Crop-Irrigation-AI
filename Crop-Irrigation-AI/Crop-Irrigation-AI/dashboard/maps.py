"""
dashboard/maps.py
─────────────────────────────────────────────────────────────────────────────
Folium-based interactive map widgets embedded in the Streamlit dashboard.
Each public `render_*` function builds a folium.Map and displays it with
streamlit-folium's st_folium().

If the actual GeoTIFF outputs don't exist yet (pipeline not run), the
functions render a placeholder map centred on the configured command area.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

try:
    import folium
    from folium import plugins as fplugins
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    import rasterio
    from rasterio.warp import transform_bounds
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from config import settings
from utils import logger

# ── Defaults (Cauvery delta command area, Tamil Nadu) ─────────────────────────
DEFAULT_LAT  = 10.85
DEFAULT_LON  = 79.15
DEFAULT_ZOOM = 11

# ── Colour maps ────────────────────────────────────────────────────────────────
CROP_PALETTE = {
    0: "#639922",   # Paddy
    1: "#BA7517",   # Maize
    2: "#1D9E75",   # Soybean
    3: "#D85A30",   # Sugarcane
    4: "#378ADD",   # Groundnut
    5: "#8E6BBF",   # Cotton
    6: "#AAAAAA",   # Fallow
}

STRESS_PALETTE = {
    0: "#1D9E75",   # No stress
    1: "#90C25A",   # Mild
    2: "#EF9F27",   # Moderate
    3: "#D85A30",   # Severe
}

DEFICIT_BREAKPOINTS = [
    (0,   "#5DCAA5"),
    (25,  "#EF9F27"),
    (50,  "#D85A30"),
    (75,  "#7B0D0D"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_map(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
              zoom: int = DEFAULT_ZOOM) -> "folium.Map":
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    fplugins.Fullscreen().add_to(m)
    fplugins.MeasureControl(position="bottomleft").add_to(m)
    return m


def _raster_bounds_center(raster_path: Path):
    """Return (lat, lon, zoom) from a raster file."""
    if not _HAS_RASTERIO or not raster_path.exists():
        return DEFAULT_LAT, DEFAULT_LON, DEFAULT_ZOOM

    with rasterio.open(raster_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    lat = (bounds[1] + bounds[3]) / 2
    lon = (bounds[0] + bounds[2]) / 2
    return lat, lon, DEFAULT_ZOOM


def _add_boundary_overlay(m: "folium.Map") -> None:
    """Add command-area boundary from shapefile if it exists."""
    if not settings.boundary_shp.exists():
        return
    try:
        import geopandas as gpd
        gdf = gpd.read_file(settings.boundary_shp).to_crs("EPSG:4326")
        folium.GeoJson(
            gdf.__geo_interface__,
            name="Command Area Boundary",
            style_function=lambda _: {
                "color": "#1D9E75",
                "weight": 2.5,
                "fillOpacity": 0.0,
                "dashArray": "6 3",
            },
            tooltip="Command Area",
        ).add_to(m)
    except Exception as exc:
        logger.warning(f"Could not load boundary: {exc}")


def _placeholder_note(label: str) -> None:
    st.info(
        f"**{label}** not yet generated.  "
        "Run the satellite pipeline from the sidebar to create output maps.",
        icon="🛰️",
    )


def _show_map(m: "folium.Map", key: str, height: int = 480):
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=height, key=key)


# ── Crop Classification Map ────────────────────────────────────────────────────

def render_crop_map(height: int = 480):
    """
    Render the crop classification raster as a colour-coded image overlay.
    Falls back to a legend card when the GeoTIFF is not available.
    """
    if not _HAS_FOLIUM:
        st.error("Install streamlit-folium: `pip install streamlit-folium`")
        return

    crop_map_path = settings.crop_map_path
    lat, lon, zoom = _raster_bounds_center(crop_map_path)
    m = _base_map(lat, lon, zoom)
    _add_boundary_overlay(m)

    if crop_map_path.exists() and _HAS_RASTERIO:
        with rasterio.open(crop_map_path) as src:
            data = src.read(1).astype(np.float32)
            bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        # Build RGBA image
        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cls, hex_col in CROP_PALETTE.items():
            mask = data == cls
            r, g, b = _hex_to_rgb(hex_col)
            rgba[mask] = [r, g, b, 200]

        import base64, io
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[bounds_4326[1], bounds_4326[0]],
                    [bounds_4326[3], bounds_4326[2]]],
            opacity=0.75,
            name="Crop Classification",
        ).add_to(m)

        # Legend
        _add_legend(m, "Crop Types", CROP_PALETTE, {
            0:"Paddy", 1:"Maize", 2:"Soybean",
            3:"Sugarcane", 4:"Groundnut", 5:"Cotton", 6:"Fallow",
        })
    else:
        _placeholder_note("Crop Classification Map")

    _show_map(m, key="crop_map", height=height)


# ── Water Deficit Map ──────────────────────────────────────────────────────────

def render_deficit_map(height: int = 480):
    """Render the water-deficit raster with a continuous colour ramp."""
    if not _HAS_FOLIUM:
        st.error("Install streamlit-folium: `pip install streamlit-folium`")
        return

    deficit_path = settings.irrigation_map_path
    lat, lon, zoom = _raster_bounds_center(deficit_path)
    m = _base_map(lat, lon, zoom)
    _add_boundary_overlay(m)

    if deficit_path.exists() and _HAS_RASTERIO:
        with rasterio.open(deficit_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata or -9999
            bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        data[data == nodata] = np.nan
        vmin, vmax = 0.0, 80.0
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)

        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        valid = ~np.isnan(data)
        for i in range(h):
            for j in range(w):
                if valid[i, j]:
                    r, g, b = _continuous_color(norm[i, j], DEFICIT_BREAKPOINTS, vmin, vmax, data[i, j])
                    rgba[i, j] = [r, g, b, 190]

        import base64, io
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[bounds_4326[1], bounds_4326[0]],
                    [bounds_4326[3], bounds_4326[2]]],
            opacity=0.75,
            name="Water Deficit (mm)",
        ).add_to(m)
    else:
        _placeholder_note("Water Deficit Map")

    _show_map(m, key="deficit_map", height=height)


# ── Moisture Stress Map ────────────────────────────────────────────────────────

def render_stress_map(height: int = 480):
    """Render the moisture-stress classification raster."""
    if not _HAS_FOLIUM:
        st.error("Install streamlit-folium: `pip install streamlit-folium`")
        return

    stress_path = settings.stress_map_path
    lat, lon, zoom = _raster_bounds_center(stress_path)
    m = _base_map(lat, lon, zoom)
    _add_boundary_overlay(m)

    if stress_path.exists() and _HAS_RASTERIO:
        with rasterio.open(stress_path) as src:
            data = src.read(1).astype(np.float32)
            bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cls, hex_col in STRESS_PALETTE.items():
            mask = data == cls
            r, g, b = _hex_to_rgb(hex_col)
            rgba[mask] = [r, g, b, 200]

        import base64, io
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[bounds_4326[1], bounds_4326[0]],
                    [bounds_4326[3], bounds_4326[2]]],
            opacity=0.75,
            name="Moisture Stress",
        ).add_to(m)

        _add_legend(m, "Moisture Stress", STRESS_PALETTE, {
            0:"No stress", 1:"Mild", 2:"Moderate", 3:"Severe",
        })
    else:
        _placeholder_note("Moisture Stress Map")

    _show_map(m, key="stress_map", height=height)


# ── Zone Advisory Map ──────────────────────────────────────────────────────────

def render_zone_advisory_map(height: int = 480):
    """
    Choropleth map of command-area zones coloured by irrigation priority.
    Uses the boundary shapefile + advisory data if available.
    """
    if not _HAS_FOLIUM:
        st.error("Install streamlit-folium: `pip install streamlit-folium`")
        return

    m = _base_map()
    _add_boundary_overlay(m)

    PRIORITY_COLORS = {
        "critical": "#D85A30",
        "moderate": "#EF9F27",
        "low":      "#90C25A",
        "adequate": "#1D9E75",
    }

    # Synthetic zone markers (replace with actual zone GeoDataFrame in production)
    synthetic_zones = [
        {"zone": "Z1", "lat": DEFAULT_LAT + 0.05, "lon": DEFAULT_LON - 0.08,
         "priority": "moderate", "deficit": 32.1, "crop": "Paddy"},
        {"zone": "Z2", "lat": DEFAULT_LAT + 0.02, "lon": DEFAULT_LON + 0.04,
         "priority": "adequate", "deficit": 12.3, "crop": "Soybean"},
        {"zone": "Z3", "lat": DEFAULT_LAT - 0.04, "lon": DEFAULT_LON + 0.09,
         "priority": "critical", "deficit": 54.2, "crop": "Maize"},
        {"zone": "Z4", "lat": DEFAULT_LAT - 0.06, "lon": DEFAULT_LON - 0.05,
         "priority": "adequate", "deficit": 9.7,  "crop": "Groundnut"},
        {"zone": "Z5", "lat": DEFAULT_LAT + 0.08, "lon": DEFAULT_LON + 0.02,
         "priority": "moderate", "deficit": 28.4, "crop": "Maize"},
        {"zone": "Z6", "lat": DEFAULT_LAT - 0.01, "lon": DEFAULT_LON - 0.12,
         "priority": "critical", "deficit": 51.8, "crop": "Sugarcane"},
    ]

    for z in synthetic_zones:
        color = PRIORITY_COLORS.get(z["priority"], "#888")
        folium.CircleMarker(
            location=[z["lat"], z["lon"]],
            radius=22,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=2,
            popup=folium.Popup(
                f"<b>{z['zone']}</b><br>"
                f"Priority: {z['priority'].upper()}<br>"
                f"Deficit: {z['deficit']} mm<br>"
                f"Crop: {z['crop']}",
                max_width=180,
            ),
            tooltip=f"{z['zone']} · {z['priority'].capitalize()}",
        ).add_to(m)

        folium.Marker(
            location=[z["lat"], z["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#fff;'
                     f'text-align:center;margin-top:6px;">{z["zone"]}</div>',
                icon_size=(44, 44),
                icon_anchor=(22, 22),
            ),
        ).add_to(m)

    _show_map(m, key="advisory_map", height=height)


# ── Internal colour utilities ──────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _continuous_color(
    norm: float,
    breakpoints: list[tuple[float, str]],
    vmin: float,
    vmax: float,
    value: float,
) -> tuple[int, int, int]:
    """Map a normalised [0,1] value to RGB via linear interpolation of breakpoints."""
    vals  = [v for v, _ in breakpoints]
    hexes = [h for _, h in breakpoints]
    rgbs  = [_hex_to_rgb(h) for h in hexes]

    # Find segment
    norm_vals = [(v - vmin) / (vmax - vmin) for v in vals]
    for i in range(len(norm_vals) - 1):
        if norm_vals[i] <= norm <= norm_vals[i + 1]:
            t = (norm - norm_vals[i]) / max(norm_vals[i + 1] - norm_vals[i], 1e-6)
            r = int(rgbs[i][0] + t * (rgbs[i + 1][0] - rgbs[i][0]))
            g = int(rgbs[i][1] + t * (rgbs[i + 1][1] - rgbs[i][1]))
            b = int(rgbs[i][2] + t * (rgbs[i + 1][2] - rgbs[i][2]))
            return r, g, b

    return _hex_to_rgb(hexes[-1])


def _add_legend(
    m: "folium.Map",
    title: str,
    palette: dict[int, str],
    labels: dict[int, str],
) -> None:
    """Inject a simple HTML legend into the folium map."""
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:14px;height:14px;background:{palette[k]};'
        f'border-radius:3px;flex-shrink:0"></div>'
        f'<span style="font-size:11px">{labels[k]}</span></div>'
        for k in sorted(palette)
        if k in labels
    )
    legend_html = f"""
    <div style="
        position:fixed; bottom:30px; left:10px; z-index:1000;
        background:rgba(255,255,255,0.93); border-radius:8px;
        padding:10px 14px; box-shadow:0 2px 8px rgba(0,0,0,0.18);
        font-family:sans-serif; min-width:140px;
    ">
      <b style="font-size:12px">{title}</b>
      <div style="margin-top:6px">{items}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
