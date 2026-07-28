"""
Tiny static dev server with HTTP Range support, so the dashboard can *seek* into the
big WAV recordings (click-to-listen) instead of downloading a whole 1-hour file per click.
Python's built-in http.server ignores Range and returns the full file; this adds 206
partial responses.

    python tools/serve.py                 # serve the repo root at http://127.0.0.1:8000/
    # then open  http://127.0.0.1:8000/site/

Serve the REPO ROOT (default) so both the page (/site/) and the recordings (/data/) are
reachable under one origin.
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + byte-range (206) support for GET/HEAD."""

    def end_headers(self):
        # A dev server: always revalidate so a rebuilt dashboard shows up on refresh.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng)
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            size = os.fstat(f.fileno()).st_size
            start = int(m.group(1)) if m and m.group(1) else 0
            end = int(m.group(2)) if m and m.group(2) else size - 1
            end = min(end, size - 1)
            if not m or start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                f.close()
                return None
            self._range = (start, end)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            f.seek(start)
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if not rng:
            return super().copyfile(source, outputfile)
        self._range = None
        remaining = rng[1] - rng[0] + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main(argv=None):
    p = argparse.ArgumentParser(description="Range-capable static server for the dashboard")
    p.add_argument("--dir", default=".", help="directory to serve (default: repo root)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="127.0.0.1")
    args = p.parse_args(argv)

    handler = partial(RangeHandler, directory=os.path.abspath(args.dir))
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"serving {os.path.abspath(args.dir)} at http://{args.bind}:{args.port}/  "
              f"(open http://{args.bind}:{args.port}/site/)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
