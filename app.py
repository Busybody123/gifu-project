"""
app.py

- app.py is in project root
- board.html / board.js / board.css and GIFs are in ./assets/
- levels are in ./levels/*.json

This app:
- loads a chosen level JSON
- embeds the board component (HTML/JS/CSS)
- passes sprite GIFs (base64) into the JS so the avatar can animate
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components


ASSETS_DIR = "assets"
LEVELS_DIR = "levels"


# ----------------------------
# Assets helpers
# ----------------------------

def asset_path(filename: str) -> str:
    return os.path.join(ASSETS_DIR, filename)


def read_text_from_assets(filename: str) -> str:
    path = asset_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_asset_file(filename: str) -> Optional[str]:
    p = asset_path(filename)
    return p if os.path.exists(p) else None


# ----------------------------
# Level discovery/loading
# ----------------------------

def discover_level_paths() -> List[str]:
    paths: List[str] = []
    if os.path.isdir(LEVELS_DIR):
        for fn in sorted(os.listdir(LEVELS_DIR)):
            if fn.lower().endswith(".json"):
                paths.append(os.path.join(LEVELS_DIR, fn))
    return paths


def load_level_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    data.setdefault("name", data["id"])
    return data


# ----------------------------
# HTML assembly
# ----------------------------

def build_board_html(level: Dict[str, Any], idle_b64: Optional[str], walk_b64: Optional[str]) -> str:
    html_tpl = read_text_from_assets("board.html")
    css = read_text_from_assets("board.css")
    js = read_text_from_assets("board.js")

    cfg = {
        "id": level["id"],
        "name": level["name"],
        "grid": level["grid"],
        "slotCount": len(level["solution"]),
        "cellPx": 44,
        "speedMs": 220,
        "maxSteps": 500,
    }

    html = html_tpl
    html = html.replace("/*__CSS__*/", css)
    html = html.replace("__CONFIG_JSON__", json.dumps(cfg))
    html = html.replace("__IDLE_GIF__", idle_b64 or "")
    html = html.replace("__WALK_GIF__", walk_b64 or "")
    html = html.replace("//__JS__//", js)

    return html


# ----------------------------
# Streamlit app
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Puzzle UI", layout="wide")
    st.title("Loop Puzzle")

    # Verify required asset files exist
    required_assets = ["board.html", "board.js", "board.css"]
    missing = [fn for fn in required_assets if not os.path.exists(asset_path(fn))]
    if missing:
        st.error(f"Missing required files in ./{ASSETS_DIR}/: {', '.join(missing)}")
        st.stop()

    # Sprites are optional; JS will fallback if missing
    idle_path = find_asset_file("p1-idle.gif")
    walk_path = find_asset_file("p2-walking.gif")
    idle_b64 = b64_file(idle_path) if idle_path else None
    walk_b64 = b64_file(walk_path) if walk_path else None

    level_paths = discover_level_paths()
    if not level_paths:
        st.error(f"No levels found. Create ./{LEVELS_DIR}/*.json")
        st.stop()

    # Load levels
    levels: List[Dict[str, Any]] = []
    errors: List[str] = []
    for p in level_paths:
        try:
            levels.append(load_level_json(p))
        except Exception as e:
            errors.append(f"{p}: {e}")

    if errors:
      for msg in errors:
        st.code(msg)

    id_to_level = {lvl["id"]: lvl for lvl in levels}

    with st.sidebar:
        st.header("Levels")
        labels = [f'{lvl["name"]} ({lvl["id"]})' for lvl in levels]
        chosen = st.selectbox("Select a level", labels, index=0)
        chosen_id = chosen.split("(")[-1].rstrip(")")
        level = id_to_level.get(chosen_id, levels[0])

        st.divider()
        st.caption(f"Assets folder: ./{ASSETS_DIR}/")

    html = build_board_html(level, idle_b64, walk_b64)
    components.html(html, height=980, scrolling=True)


if __name__ == "__main__":
    main()
