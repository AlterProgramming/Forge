"""Serve the Avatar Forge Hair Texture Lab locally.

Usage:
    python -m tools.studio
    python -m tools.studio --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/viewer/texture-lab.html"
    print(f"[hair-studio] serving {ROOT}")
    print(f"[hair-studio] {url}")
    print("[hair-studio] generate a GLB, then drag it onto the viewer")
    if not args.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
