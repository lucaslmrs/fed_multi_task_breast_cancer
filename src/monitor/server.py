"""HTTP server for the monitor: JSON API over the readers plus the static page.

Read-only and local by default (binds 127.0.0.1). Every path a request names is resolved under
the runs root and refused if it escapes it.
"""

import gzip
import json
import logging
import mimetypes
import re
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.monitor import catalog, compare, readers

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html", "app.js", "style.css"}
# Static reports and what they embed; nothing else under the runs root is served.
STUDY_ARM = re.compile(r"^studies/(.+)/runs/seed_([^/]+)/([^/]+)$")
SERVED_SUFFIXES = {".html", ".png", ".svg", ".jpg", ".jpeg", ".css", ".js", ".csv", ".json",
                   ".md", ".txt"}


class MonitorApp:
    def __init__(self, runs_root, refresh_seconds=10, stale_minutes=catalog.STALE_MINUTES,
                 baseline_root=None):
        self.root = Path(runs_root).resolve()
        self.refresh_seconds = refresh_seconds
        self.stale_minutes = stale_minutes
        self.baseline_root = Path(baseline_root) if baseline_root else self.root / "baseline"

    def scan(self):
        return catalog.scan(self.root, self.stale_minutes)

    def run_dir(self, rel_path):
        if not rel_path:
            raise FileNotFoundError("missing 'path'")
        target = catalog.resolve(self.root, rel_path)
        if not catalog.is_run_dir(target):
            raise FileNotFoundError(f"{rel_path} is not a run directory")
        return target

    def entry(self, run_dir):
        rel = catalog.relative(run_dir, self.root)
        group = rel.split("/", 1)[0]
        labels = {}
        arm = STUDY_ARM.match(rel)
        if arm:   # same labels the catalog gives a study arm, so names agree everywhere
            group = "study"
            labels = {"study_id": arm.group(1), "seed": arm.group(2), "arm_id": arm.group(3)}
        elif group not in ("baseline", "validation") or rel == group:
            group = "runs"
        return asdict(catalog.describe_run(run_dir, self.root, group, labels,
                                           stale_minutes=self.stale_minutes))


class Handler(BaseHTTPRequestHandler):
    app: MonitorApp = None
    server_version = "fedmonitor/1"

    def log_message(self, format, *args):  # noqa: A002 - signature from the base class
        logging.debug("%s - %s", self.address_string(), format % args)

    # ---- responses ----
    def _send(self, status, body, content_type):
        if "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 2048:
            body = gzip.compress(body, compresslevel=5)
            encoding = "gzip"
        else:
            encoding = None
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(readers.clean(payload), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",
                                                                "application/json"):
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    # ---- routing ----
    def do_GET(self):  # noqa: N802 - name from the base class
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = unquote(url.path)
        try:
            if method == "GET" and path in ("/", "/index.html"):
                return self._file(STATIC_DIR / "index.html")
            if method == "GET" and path.startswith("/static/"):
                name = path[len("/static/"):]
                if name not in STATIC_FILES:
                    return self._error(HTTPStatus.NOT_FOUND, "unknown static file")
                return self._file(STATIC_DIR / name)
            if method == "GET" and path.startswith("/files/"):
                return self._serve_run_file(path[len("/files/"):])
            if path.startswith("/api/"):
                return self._api(method, path[len("/api/"):], query)
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except PermissionError as error:
            return self._error(HTTPStatus.FORBIDDEN, str(error))
        except FileNotFoundError as error:
            return self._error(HTTPStatus.NOT_FOUND, str(error))
        except Exception as error:  # keep the server alive on a malformed artifact
            logging.exception("monitor request failed: %s", self.path)
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                               f"{type(error).__name__}: {error}")

    def _serve_run_file(self, rel_path):
        target = catalog.resolve(self.app.root, rel_path)
        if target.suffix.lower() not in SERVED_SUFFIXES:
            raise PermissionError(f"{target.suffix} files are not served")
        if not target.is_file():
            raise FileNotFoundError(rel_path)
        return self._file(target)

    def _api(self, method, name, query):
        app = self.app
        first = (query.get("path") or [""])[0]
        if method == "POST":
            if name != "diagnose":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "only /api/diagnose is POST")
            return self._json(readers.diagnose_now(app.run_dir(first), app.baseline_root))
        if name == "config":
            return self._json({"root": str(app.root), "refresh_seconds": app.refresh_seconds,
                               "stale_minutes": app.stale_minutes,
                               "task_modes": catalog.TASK_MODES})
        if name == "runs":
            return self._json(app.scan())
        if name == "run":
            run_dir = app.run_dir(first)
            return self._json({"entry": app.entry(run_dir),
                               "final": readers.final_metrics(run_dir)})
        if name == "compare":
            paths = query.get("path") or []
            if len(paths) < 2:
                return self._error(HTTPStatus.BAD_REQUEST, "compare needs at least two 'path'")
            run_dirs = [app.run_dir(p) for p in paths]
            return self._json(compare.compare(run_dirs, [app.entry(d) for d in run_dirs]))
        readers_by_name = {"aggregation": readers.aggregation, "conflict": readers.conflict,
                           "resources": readers.resources, "diagnostics": readers.diagnostics}
        if name == "curves":
            fold = (query.get("fold") or [None])[0]
            return self._json(readers.curves(app.run_dir(first), fold or None))
        if name in readers_by_name:
            return self._json(readers_by_name[name](app.run_dir(first)))
        return self._error(HTTPStatus.NOT_FOUND, f"unknown endpoint {name}")


def make_server(runs_root, host="127.0.0.1", port=8765, **kwargs):
    handler = type("BoundHandler", (Handler,), {"app": MonitorApp(runs_root, **kwargs)})
    return ThreadingHTTPServer((host, port), handler)
