"""Launch the browser interface or add an older generation to its history."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

from ..run_history import ROLES, TIERS, RunHistory
from .._locking import LockBusy, process_lock
from .server import WebRuns, make_server


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Select a port between 1 and 65535.") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Select a port between 1 and 65535.")
    return port


def _serve(history: RunHistory, port: int, open_browser: bool) -> None:
    with process_lock(history.root / ".web-server.lock"):
        runs = WebRuns(history)
        runs.recover()
        with make_server(runs, port) as server:
            url = f"http://127.0.0.1:{port}/"
            print(f"LocalMesh Engine is listening at {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                runs.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m localmesh_engine.web", description=__doc__)
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="start a local web server")
    serve.add_argument("--port", type=_port, default=8765, help="loopback port (default: 8765)")
    serve.add_argument("--legacy-dir", type=Path, default=Path("out"),
                       help="read existing CLI outputs from this directory (default: out)")
    serve.add_argument("--open", action="store_true", help="open the web interface in your default browser")
    importer = commands.add_parser("import", help="archive a GLB and its original source photos")
    importer.add_argument("model", type=Path, help="path to the main GLB file")
    for role in ROLES:
        importer.add_argument(f"--{role}", required=role == "front", type=Path,
                              metavar="IMAGE", help=f"original {role} image")
    importer.add_argument("--tier", choices=TIERS, help="quality tier used to generate the model")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0].split("=", 1)[0] in ("--port", "--legacy-dir", "--open"):
        raw_args.insert(0, "serve")
    args = parser.parse_args(raw_args)

    try:
        from .. import config

        if args.command == "import":
            sources = {role: (image.name, image) for role in ROLES
                       if (image := getattr(args, role)) is not None}
            row = RunHistory(config.RUNTIME_ROOT).import_run(args.model, sources, tier=args.tier)
            print(f"Imported {args.model.name} into LocalMesh history ({row['id']}).")
        else:
            _serve(RunHistory(config.RUNTIME_ROOT, args.legacy_dir), args.port, args.open)
    except LockBusy:
        print("A LocalMesh web server is already using this runtime.", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
