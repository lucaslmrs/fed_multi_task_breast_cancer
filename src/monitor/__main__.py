"""``python -m src.monitor``: serve the local monitoring app."""

import argparse
import logging

from src.monitor.catalog import STALE_MINUTES
from src.monitor.server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local, read-only training monitor (plano E2).")
    parser.add_argument("--runs", default="runs", help="runs root to browse (default: runs)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; keep the default unless you know you need another")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=float, default=10.0,
                        help="polling period of a running run in the page, in seconds")
    parser.add_argument("--stale-minutes", type=float, default=STALE_MINUTES,
                        help="a run without results and idle longer than this is 'stale'")
    parser.add_argument("--baseline-root", default=None,
                        help="baselines used by on-demand diagnostics (default: <runs>/baseline)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = make_server(args.runs, args.host, args.port, refresh_seconds=args.refresh,
                         stale_minutes=args.stale_minutes, baseline_root=args.baseline_root)
    host, port = server.server_address[:2]
    logging.info("Training monitor on http://%s:%s/  (runs root: %s; Ctrl+C to stop)", host, port,
                 server.RequestHandlerClass.app.root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
