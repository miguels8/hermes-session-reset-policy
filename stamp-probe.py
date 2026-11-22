#!/usr/bin/env python3
"""Prove a reset-policy clock stamp persists ONE routing row, never the index.

Alert sig:system|e001fd8d32: the session-reset-policy hook stamped its clock on
every inbound Discord message through set_session_metadata -> _update_entry ->
_save(), which serialized and fsynced the whole gateway_routing index plus the
sessions.json mirror on the gateway thread; Discord logged "heartbeat blocked
for more than 10 seconds" and then "Attempting a reconnect".

Runs against the tree and plugin named by CL_TREE / CL_PLUGIN (defaults: the
live gateway tree and the live plugin). Exit 0 only when

  * the plugin's _stamp_clock takes the engine's single-row, identity-checked
    write and the full-index replacer is never called,
  * the value lands in state.db and survives a store reload,
  * a route rotated before the write is left unstamped: the stale stamp is
    refused and the replacement's own row, read back from SQLite, carries no
    clock (a plugin without the route guard fails here; the plugin's test suite
    runs this probe against such a plugin and requires PROBE RED).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile

TREE = os.environ.get("CL_TREE", "/home/santo/dev/.worktrees/hermes-agent/update-20260917")
PLUGIN = os.environ.get(
    "CL_PLUGIN", "/home/santo/.hermes/plugins/hermes-session-reset-policy/__init__.py"
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    sys.path.insert(0, TREE)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stamp-probe-"))
    import hermes_constants
    import hermes_state

    hermes_state.DEFAULT_DB_PATH = tmp / "state.db"
    hermes_constants.get_hermes_home = lambda: str(tmp)

    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    spec = importlib.util.spec_from_file_location("srp_probe", PLUGIN)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    def make_store() -> SessionStore:
        return SessionStore(sessions_dir=tmp / "sessions", config=GatewayConfig())

    source = SessionSource(
        platform=Platform.LOCAL, chat_id="cli", chat_name="CLI", chat_type="dm",
        user_id="user-1",
    )

    store = make_store()
    entry = store.get_or_create_session(source)
    routing_db = store._routing_db
    calls = []
    real = routing_db.replace_gateway_routing_entries

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    routing_db.replace_gateway_routing_entries = spy

    stamp = getattr(plugin, "_stamp_clock")
    ok = stamp(store, entry.session_key, entry.session_id, "2026-11-21T07:31:18")
    check("stamp reports the clock advanced", ok is True, f"returned {ok!r}")
    check("full-index rewrite not called", not calls, f"{len(calls)} call(s)")

    rows = store._db.load_gateway_routing_entries(scope=store._routing_scope())
    row = json.loads(rows[entry.session_key])
    check(
        "value persisted in state.db",
        row.get("metadata", {}).get(plugin._META_KEY) == "2026-11-21T07:31:18",
        str(row.get("metadata")),
    )
    check("activity clock not advanced", row["updated_at"] == entry.updated_at.isoformat())
    store._db.close()

    restarted = make_store()
    check(
        "value rebinds after a store restart",
        restarted.get_session_metadata(entry.session_key, plugin._META_KEY)
        == "2026-11-21T07:31:18",
    )

    # A stamp for a route that already rotated belongs to the live route; the
    # plugin binds its own clock to the session id it saw, so it can never read
    # the replacement's stamp as the old conversation's activity.
    before = restarted.get_or_create_session(source)
    stalled = before.session_id
    rotated = restarted.get_or_create_session(source, force_new=True)
    restarted._db.replace_gateway_routing_entries = spy
    calls.clear()
    refused = stamp(restarted, rotated.session_key, stalled, "2026-11-21T07:32:00")
    check("rotation took no whole-index write", not calls, f"{len(calls)} call(s)")
    check("stale stamp is refused", refused is False, f"returned {refused!r}")
    # The proof is in the replacement's own row in state.db, read back from SQLite:
    # a plugin that lost its route guard stamps it, and a local cache cannot show that.
    rows = restarted._db.load_gateway_routing_entries(scope=restarted._routing_scope())
    replacement = json.loads(rows[rotated.session_key])
    check(
        "SQLite row is the replacement route",
        replacement.get("session_id") == rotated.session_id,
        f"row names {replacement.get('session_id')!r}, expected {rotated.session_id!r}",
    )
    check(
        "replacement metadata untouched in SQLite",
        plugin._META_KEY not in (replacement.get("metadata") or {}),
        str(replacement.get("metadata")),
    )
    check(
        "local clock is bound to the route it saw",
        plugin._local_clock(rotated.session_key, stalled) is not None
        and plugin._local_clock(rotated.session_key, rotated.session_id) is None,
    )
    restarted._db.close()

    print(f"tree={TREE}")
    print(f"plugin={PLUGIN}")
    print("PROBE " + ("GREEN" if not failures else "RED: " + ", ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
