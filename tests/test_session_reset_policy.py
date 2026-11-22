"""Tests for the hermes-session-reset-policy plugin.

Standalone-suite discipline (per the platform-plugin skill):
- pytest must be able to COLLECT this file without any Hermes import
  succeeding. All hermes imports are therefore behind helpers inside tests,
  never at module scope.
- Real classes from the installed Hermes tree are imported lazily inside
  helpers, so the tests exercise the actual SessionStore/SessionEntry
  behavior (routing transitions, reset_session) rather than mocks.
- The reset path has two layers (review #2): the gateway's /new reset funnel
  (``gateway._handle_reset_command``) when reachable, and a degraded bare
  store rotation otherwise. Both are covered; the funnel layer additionally
  runs the REAL mixin body bound to a minimal runner stub.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"
def _hermes_tree() -> Path:
    """The Hermes checkout the real-engine tests import.

    ``SRP_HERMES_TREE`` lets an evidence run point at the exact candidate
    worktree; without it, the installed tree for this user. On this host that
    path does not exist (the canonical checkout is ``/home/santo/dev``), so the
    real-engine tests skip unless the override is set — the standalone seam
    tests still run.
    """
    override = os.environ.get("SRP_HERMES_TREE")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "hermes-agent"


HERMES_TREE = _hermes_tree()


def _run(coro):
    """Drive the (async) hook body to completion; the suite runs without pytest-asyncio."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _temp_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME for every test.

    The real SessionStore opens <hermes_home>/state.db; Hermes' live-system
    guard correctly refuses any test-context open of the production DB.
    A temporary HERMES_HOME is resolved at call time (hermes_state._default_db_path),
    so the redirect reaches session storage regardless of import order, and
    the guard passes because the temp root is not the platform default.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    # Hermetic root inference (issue #4 repro exposed this): on hosts whose
    # TMPDIR sits under the native ~/.hermes (e.g. the Hermes scratch dir),
    # pytest tmp roots read as part of the DEFAULT profile tree — root
    # inference then anchors at ~/.hermes and profile-name resolution
    # (hermes_cli.profiles.get_profile_dir) escapes the sandbox into the LIVE
    # profiles/ dir. Shift the platform-default suffix so every tmp root is a
    # self-contained external root.
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", f"/pytest-{tmp_path.name}")
    try:
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)
    except Exception:
        pass  # tree not importable here: the tests skip anyway


@pytest.fixture(autouse=True)
def hook_recorder(monkeypatch, _temp_hermes_home):
    """Record — never execute — the boundary side effects the reset path fires
    through host modules: ``hermes_cli.lifecycle.invoke_hook`` (the
    on_session_finalize / on_session_reset plugin hooks) and
    ``tools.async_delegation.interrupt_for_session``. Both are patched on the
    source module, and the plugin imports them lazily at call time, so the
    recorders intercept every path (funnel tail and degraded fallback)."""
    rec = SimpleNamespace(lifecycle=[], interrupts=[])
    if HERMES_TREE.is_dir():
        sys.path.insert(0, str(HERMES_TREE))
    try:
        import hermes_cli.lifecycle as lifecycle
        monkeypatch.setattr(
            lifecycle, "invoke_hook",
            lambda name, **kw: rec.lifecycle.append((name, kw)) or [])
    except Exception:
        pass  # tree not importable here: the plugin's own try/suppress covers it
    try:
        import tools.async_delegation as async_delegation
        monkeypatch.setattr(
            async_delegation, "interrupt_for_session",
            lambda **kw: rec.interrupts.append(kw) or 0)
    except Exception:
        pass
    return rec


@pytest.fixture()
def plugin():
    """Load the plugin module standalone (spec_from_file_location shape)."""
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.hermes_session_reset_policy", PLUGIN_INIT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _import_tree_module(name: str) -> Any:
    """Import *name* from the installed Hermes tree, or skip the test."""
    if not HERMES_TREE.is_dir():
        pytest.skip("Hermes tree not present on this host")
    sys.path.insert(0, str(HERMES_TREE))
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"Hermes tree module not importable: {name}: {exc}")


# ---------------------------------------------------------------------------
# _time_reset_reason — boundary math (ported verbatim from the core patch)
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 26, 12, 0, 0)


def _last(updated_minutes_ago):
    """A user-activity clock value N minutes before NOW."""
    if updated_minutes_ago is None:
        return None
    return NOW - timedelta(minutes=updated_minutes_ago)

def _entry(updated_minutes_ago: float, *, token=None):
    from dataclasses import dataclass, field

    @dataclass
    class Entry:
        created_at: datetime = NOW - timedelta(minutes=updated_minutes_ago or 1)
        updated_at: datetime = NOW - timedelta(minutes=updated_minutes_ago or 1)
        active_turn_token: str | None = token
        session_key: str = "agent:main:local:dm:test"
        session_id: str = "s-1"

    return Entry()


def test_idle_mode_triggers_past_threshold(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(90), NOW) == "idle"

def test_idle_mode_no_reset_within_threshold(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(30), NOW) is None

def test_idle_mode_exactly_at_threshold_triggers(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(60), NOW) == "idle"

def test_daily_mode_boundary(plugin):
    # now=12:00, boundary hour=4 -> boundary today 04:00; last= yesterday -> reset
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 10, 0),
        updated_at=datetime(2026, 9, 25, 10, 0),
    )
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 25, 10, 0), NOW) == "daily"
    # last after today's boundary -> no reset
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 8, 0), NOW) is None

def test_daily_mode_boundary_in_future_rolls_back_a_day(plugin):
    # now=02:00, boundary hour=4 (today 04:00 is in the future) -> boundary
    # yesterday 04:00; last=yesterday 12:00 (after it) -> no reset.
    now = datetime(2026, 9, 26, 2, 0)
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 12, 0),
        updated_at=datetime(2026, 9, 25, 12, 0),
    )
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 25, 12, 0), now) is None
    # last=day-before-yesterday -> reset
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 24, 12, 0), now) == "daily"

def test_both_mode_idle_wins_first(plugin):
    policy = {"mode": "both", "idle_minutes": 30, "at_hour": 4}
    assert plugin._time_reset_reason(policy, _last(90), NOW) == "idle"

def test_none_mode_never_triggers(plugin):
    policy = {"mode": "none"}
    assert plugin._time_reset_reason(policy, _last(100000), NOW) is None

def test_malformed_policy_fails_open(plugin):
    for policy in ({"mode": "idle", "idle_minutes": "lots"}, {"mode": "daily", "at_hour": 99}, {}):
        assert plugin._time_reset_reason(policy, _last(90), NOW) is None

def test_missing_timestamps_no_reset(plugin):
    policy = {"mode": "idle", "idle_minutes": 1}
    assert plugin._time_reset_reason(policy, None, NOW) is None

def test_created_at_fallback(plugin):
    # Migration path: entry without metadata clock falls back to created_at
    policy = {"mode": "idle", "idle_minutes": 60}
    entry = SimpleNamespace(
        created_at=NOW - timedelta(minutes=90),
        updated_at=None,
        metadata={},
    )
    assert plugin._time_reset_reason(policy, plugin._last_user_inbound(entry), NOW) == "idle"


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

def test_register_registers_hook(plugin):
    ctx = mock.MagicMock()
    plugin.register(ctx)
    ctx.register_hook.assert_called_once_with("pre_gateway_dispatch", plugin._maybe_reset)


# ---------------------------------------------------------------------------
# _command_free_event — the funnel must never title the session from the
# triggering utterance (/new <title> semantics; get_command_args() on a plain
# message is the whole message text).
# ---------------------------------------------------------------------------

def test_reset_event_strips_message_text(plugin):
    from dataclasses import dataclass

    @dataclass
    class Event:
        source: object = "src"
        text: str = "please start fresh tomorrow morning"
        def get_command_args(self) -> str:
            return self.text  # non-command MessageEvent semantics

    event = Event()
    bare = plugin._command_free_event(event)
    assert bare.get_command_args() == ""
    assert bare.source is event.source
    assert event.text == "please start fresh tomorrow morning"  # original untouched

def test_reset_event_falls_back_for_plain_objects(plugin):
    source = object()
    event = SimpleNamespace(source=source, get_command_args=lambda: "hello there")
    bare = plugin._command_free_event(event)
    assert bare.get_command_args() == ""
    assert bare.source is source


# ---------------------------------------------------------------------------
# _maybe_reset — end-to-end against the REAL SessionStore from the tree
# ---------------------------------------------------------------------------

def _make_real_store(tmp_path):
    """Build a real SessionStore over a scratch sessions dir."""
    mod = _import_tree_module("gateway.session")
    cfg = _import_tree_module("gateway.config")
    store = mod.SessionStore(sessions_dir=tmp_path / "sessions", config=cfg.GatewayConfig())
    return store


def _real_source(chat_id="test-chat"):
    mod = _import_tree_module("gateway.session")
    cfg = _import_tree_module("gateway.config")
    return mod.SessionSource(platform=cfg.Platform.DISCORD, chat_id=chat_id, chat_type="dm",
                             user_id="tester")

def _real_event(text="please keep going", chat_id="test-chat"):
    """A real inbound MessageEvent (it carries ``accept_interrupt``, which the
    gateway's reset handler calls), not a stand-in namespace."""
    event_mod = _import_tree_module("gateway.platforms.event")
    return event_mod.MessageEvent(text=text, source=_real_source(chat_id))

def _seed_route(store, *, chat_id="test-chat", updated_minutes_ago=0.0, token=None):
    """Open a genuine route through the engine's own door, then age its clock.

    ``get_or_create_session`` is what a first message calls, so the route carries
    every record the engine's reset path reads. The old hand-built entry skipped
    them and, being a LOCAL route, every rotation died on a missing local
    authority epoch: a LOCAL route needs a booted gateway, and the plugin's
    traffic is messaging platforms, so the route is a Discord one. The user
    clock is back-dated through the engine's own setter (the plugin's real
    store), and ``token`` marks a turn in flight. Returns ``(key, entry)``; the
    session id is the engine's, not a constant.
    """
    entry = store.get_or_create_session(_real_source(chat_id))
    key = entry.session_key
    assert store.lookup_by_session_key(key) is entry
    if updated_minutes_ago:
        aged = datetime.now() - timedelta(minutes=updated_minutes_ago)
        assert store.set_session_metadata(key, "srp_last_user_inbound", aged.isoformat())
    entry.active_turn_token = token
    return key, entry


def test_maybe_reset_store_fallback_calls_real_reset_session(plugin, tmp_path, hook_recorder):
    """Degraded path (gateway double has no reset funnel): a bare store rotation
    PLUS what the plugin surface can reach — the delegation interrupt and the
    on_session_finalize / on_session_reset lifecycle pair."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        result = _run(plugin._maybe_reset(event, gateway, store))

    assert result is None  # allow normal dispatch
    new_entry = store.lookup_by_session_key(key)
    assert new_entry is not None
    assert new_entry.session_id != entry.session_id
    assert new_entry.is_fresh_reset is True
    # In-flight delegations end with the old conversation (review #2, item 3).
    assert hook_recorder.interrupts
    assert hook_recorder.interrupts[0]["session_key"] == key
    assert hook_recorder.interrupts[0]["parent_session_id"] == entry.session_id
    # Boundary hooks fire (review #2, item 2): finalize for the old id, reset for the new.
    fired = [name for name, _ in hook_recorder.lifecycle]
    assert "on_session_finalize" in fired
    assert "on_session_reset" in fired
    reset_kw = dict(hook_recorder.lifecycle)["on_session_reset"]
    assert reset_kw["old_session_id"] == entry.session_id
    assert reset_kw["new_session_id"] == new_entry.session_id

def test_maybe_reset_prefers_the_gateway_funnel(plugin, tmp_path, hook_recorder):
    """When the gateway reset funnel exists it IS the reset — the store is never
    rotated behind its back — and it receives a command-free event so /new's
    title-from-args step cannot title the fresh session from the utterance."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    seen = []

    async def funnel(evt):
        seen.append(evt)

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False,
                            get_command_args=lambda: "and please keep the old context in mind")

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy), \
         mock.patch.object(store, "reset_session",
                           side_effect=AssertionError("store must not rotate behind the funnel")):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert len(seen) == 1
    assert seen[0].get_command_args() == ""  # no title can leak in as /new args
    assert seen[0].source is event.source

def test_maybe_reset_funnel_failure_before_rotation_falls_back(plugin, tmp_path, hook_recorder):
    """A funnel that blows up before rotating degrades to the store rotation;
    the boundary must still happen."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)

    async def funnel(evt):
        raise RuntimeError("funnel exploded")

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry.session_id != entry.session_id
    assert new_entry.is_fresh_reset is True
    assert hook_recorder.interrupts  # fallback still ends old delegations

def test_maybe_reset_funnel_failure_after_rotation_keeps_result(plugin, tmp_path, hook_recorder):
    """A funnel that rotates and THEN blows up (banner/tip tail) must not trigger
    a second rotation in the fallback."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    rotated_sid = []

    async def funnel(evt):
        rotated_sid.append(store.reset_session(key).session_id)
        raise RuntimeError("banner exploded after the rotation")

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == rotated_sid[0]

def test_maybe_reset_prefers_payload_session_key(plugin, tmp_path):
    """A public session_key in the hook payload outranks the private gateway
    helper (review #2, item 4: ready for the payload equivalent they offered)."""
    store = _make_real_store(tmp_path)
    payload_key, _ = _seed_route(store, chat_id="payload-chat")
    helper_key, _ = _seed_route(store, chat_id="helper-chat")
    assert payload_key != helper_key
    gateway = SimpleNamespace(_session_key_for_source=lambda src: helper_key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}  # within threshold -> clock-stamp path
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store,
                                        session_key=payload_key)) is None

    assert store.get_session_metadata(payload_key, "srp_last_user_inbound") is not None
    assert store.get_session_metadata(helper_key, "srp_last_user_inbound") is None

def test_maybe_reset_bg_turn_does_not_block_reset(plugin, tmp_path, hook_recorder):
    """The 19:04 bug: a background-review turn stamps ``updated_at`` but must
    NOT reset the plugin's own clock. An old last-user-inbound with a fresh
    ``updated_at`` must still reset."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=1)  # fresh updated_at
    # But the plugin's clock says the last real user inbound was 2h ago:
    store.set_session_metadata(key, "srp_last_user_inbound",
                               (datetime.now() - timedelta(hours=2)).isoformat())

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry.session_id != entry.session_id
    assert new_entry.is_fresh_reset is True

def test_maybe_reset_stamps_clock_when_within_threshold(plugin, tmp_path):
    """Within the threshold the message must advance the plugin's own clock,
    not ``updated_at``."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    before = datetime.now() - timedelta(seconds=5)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == entry.session_id
    stamp = store.get_session_metadata(key, "srp_last_user_inbound")
    assert stamp is not None
    assert datetime.fromisoformat(stamp) >= before

def test_maybe_reset_deferral_does_not_advance_clock(plugin, tmp_path):
    """Boundary past + turn in flight: defer AND leave the clock alone, so the
    next message re-evaluates the same boundary."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="tok-123")
    seeded_clock = store.get_session_metadata(key, "srp_last_user_inbound")
    assert seeded_clock is not None
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == entry.session_id
    # No clock stamp: the deferral must not swallow the pending reset.
    assert store.get_session_metadata(key, "srp_last_user_inbound") == seeded_clock

def test_maybe_reset_skips_when_within_threshold(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == entry.session_id

def test_maybe_reset_defers_when_turn_in_flight(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="tok-123")
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    # Route unchanged: the in-flight turn keeps its session.
    assert store.lookup_by_session_key(key).session_id == entry.session_id

def test_maybe_reset_no_policy_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=100000)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=None):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    assert store.lookup_by_session_key(key).session_id == entry.session_id

def test_maybe_reset_survives_store_errors(plugin, tmp_path):
    class ExplodingStore:
        def lookup_by_session_key(self, key):
            raise RuntimeError("boom")

    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:x")
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, ExplodingStore())) is None

def test_maybe_reset_no_route_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:none")
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

def test_maybe_reset_survives_missing_key_helper(plugin, tmp_path):
    """No payload key and no gateway helper: fail closed (no reset, no stamp).
    A bare build_session_key fallback is deliberately absent — under
    multiplexed profiles it is not profile-namespaced and could resolve
    ANOTHER profile's route (review #2, item 4)."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gateway = SimpleNamespace()  # no _session_key_for_source at all
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    assert store.lookup_by_session_key(key).session_id == entry.session_id


# ---------------------------------------------------------------------------
# The REAL /new reset funnel — GatewaySessionCommandsMixin bodies bound to a
# minimal runner stub, run against the REAL SessionStore (review #2, items 1-3:
# run-generation bump, cached-agent eviction, conversation-scope clear,
# async-delegation interrupt, session:end/session:reset, boundary hooks).
# ---------------------------------------------------------------------------

def _real_funnel_gateway(store, key):
    """A GatewayRunner stand-in carrying the REAL funnel methods.

    Only the leaf internals the funnel touches are recorders (they are the
    gateway's own per-session caches/hooks, covered by upstream tests);
    ``_handle_reset_command``, ``_fire_session_reset_hooks`` and
    ``_cleanup_old_agent_for_reset`` run for real."""
    session_mod = _import_tree_module("gateway.session")
    slash_mod = _import_tree_module("gateway.slash_commands_session")
    mixin = slash_mod.GatewaySessionCommandsMixin

    calls = SimpleNamespace(invalidated=[], released=[], evicted=[], cleared=[], finalize=[],
                            stopped=[], delivered=[])

    class _Hooks:
        def __init__(self):
            self.emitted = []

        async def emit(self, name, payload):
            self.emitted.append((name, payload))

    async def _finalize_session_off_loop(**kw):
        calls.finalize.append(kw)

    async def _hm_stop_owner_turn(session_key):
        calls.stopped.append(session_key)

    async def _reset_session_off_loop(*, session_key=None, **kwargs):
        # The engine runs this body in an executor under the owning profile's
        # scope; the body itself is the one plugin hook call.
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook("on_session_reset", **kwargs)

    async def _deliver_session_end_messages(messages, **kw):
        calls.delivered.append(messages)

    gw = SimpleNamespace()
    gw.session_store = store
    gw.async_session_store = session_mod.AsyncSessionStore(store)
    gw.hooks = _Hooks()
    gw._session_key_for_source = lambda src: key
    gw._cached_agent_for = lambda k: None  # real cleanup helper early-returns on None
    gw._invalidate_session_run_generation = lambda k, reason="": calls.invalidated.append((k, reason))
    gw._release_running_agent_state = lambda k: calls.released.append(k)
    gw._evict_cached_agent = lambda k: calls.evicted.append(k)
    gw._clear_conversation_scope = lambda k, reason="": calls.cleared.append((k, reason))
    gw._finalize_session_off_loop = _finalize_session_off_loop
    gw._hm_stop_owner_turn = _hm_stop_owner_turn
    gw._reset_session_off_loop = _reset_session_off_loop
    gw._reset_notice_session_info = lambda source: ""
    gw._deliver_session_end_messages = _deliver_session_end_messages
    gw._async_profile_scope_for_source = lambda source: contextlib.nullcontext()
    gw._telegram_topic_new_header = lambda source: ""
    gw._is_telegram_topic_lane = lambda source: False
    gw._session_db = None
    for name in ("_handle_reset_command", "_fire_session_reset_hooks",
                 "_fire_plugin_session_reset", "_cleanup_old_agent_for_reset"):
        setattr(gw, name, getattr(mixin, name).__get__(gw))
    return gw, calls, slash_mod


def test_maybe_reset_runs_the_real_new_funnel(plugin, tmp_path, hook_recorder, monkeypatch):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gw, calls, slash_mod = _real_funnel_gateway(store, key)
    monkeypatch.setattr(slash_mod, "t", lambda key_name, **kw: key_name)  # no i18n in tests
    event = _real_event()  # the real handler calls event.accept_interrupt()

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gw, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry is not None
    assert new_entry.session_id != entry.session_id
    assert new_entry.is_fresh_reset is True
    # Item 1: run-generation bump, running-slot release, cached-agent eviction,
    # conversation-scope clear — the exact /new boundary sequence.
    assert calls.invalidated == [(key, "session_reset")]
    assert calls.released == [key]
    assert calls.evicted == [key]
    assert calls.cleared == [(key, "session_reset")]
    # Item 2: gateway session:end / session:reset hooks + on_session_finalize.
    assert [name for name, _ in gw.hooks.emitted] == ["session:end", "session:reset"]
    assert gw.hooks.emitted[0][1]["session_key"] == key
    assert calls.finalize and calls.finalize[0]["session_id"] == entry.session_id
    assert calls.finalize[0]["new_session_id"] == new_entry.session_id
    fired = dict(hook_recorder.lifecycle)
    assert "on_session_reset" in fired
    assert fired["on_session_reset"]["old_session_id"] == entry.session_id
    assert fired["on_session_reset"]["new_session_id"] == new_entry.session_id
    # Item 3: in-flight async delegations end with the old conversation.
    assert hook_recorder.interrupts
    assert hook_recorder.interrupts[0]["session_key"] == key
    assert hook_recorder.interrupts[0]["parent_session_id"] == entry.session_id


# ---------------------------------------------------------------------------
# _read_session_reset_policy — profile resolution via the real config loader
# ---------------------------------------------------------------------------

def _bootstrap_hermes_paths():
    sys.path.insert(0, str(HERMES_TREE))
    return HERMES_TREE.is_dir()

def test_policy_reader_none_on_missing_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    import hermes_constants
    from hermes_cli import config as cfgmod

    # Point the config cache at the temp home (the autouse fixture redirects
    # HERMES_HOME already; ensure no stale cache survives from other tests).
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)

    assert plugin._read_session_reset_policy(None) is None

def test_policy_reader_reads_named_profile_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    from hermes_cli import config as cfgmod
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    prof = home / "profiles" / "mom"
    prof.mkdir(parents=True)
    (prof / "config.yaml").write_text(
        "session_reset:\n  mode: both\n  idle_minutes: 45\n  at_hour: 4\n"
    )

    policy = plugin._read_session_reset_policy("mom")
    assert policy == {"mode": "both", "idle_minutes": 45, "at_hour": 4}

def test_policy_reader_reads_default_profile_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    from hermes_cli import config as cfgmod
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "session_reset:\n  mode: idle\n  idle_minutes: 30\n"
    )

    policy = plugin._read_session_reset_policy(None)
    assert policy == {"mode": "idle", "idle_minutes": 30}

def _write_idle_policy(config_dir: Path, minutes: int) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        f"session_reset:\n  mode: idle\n  idle_minutes: {minutes}\n", encoding="utf-8"
    )

def test_policy_reader_resolves_profiles_from_root(plugin, tmp_path, monkeypatch):
    """Issue #4 regression: profile tags resolve against the Hermes ROOT under
    BOTH gateway layouts — a multiplexed gateway (HERMES_HOME = <root>) and a
    standalone per-profile gateway (HERMES_HOME = <root>/profiles/<name>). The
    old ``<home>/profiles/<name>`` build double-nested on the standalone layout
    (looking for ``profiles/alpha/profiles/alpha/config.yaml``) and the reader
    silently returned None, so no reset ever fired.

    ``default`` is just a profile name too: its home IS the root, whatever the
    launch home is (same mapping hermes_cli.profiles.get_profile_dir encodes).
    """
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    import hermes_constants
    from hermes_cli import config as cfgmod

    # External/custom-root layout (the reporter's per-profile gateways, cf. a
    # Docker /opt/data root): shift the native-home suffix so the tmp root is
    # not absorbed into the default profile tree's root inference.
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", f"/srp-test-{tmp_path.name}")
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)

    root = tmp_path / "root"
    _write_idle_policy(root, 111)                    # the "default" profile
    _write_idle_policy(root / "profiles" / "alpha", 222)
    _write_idle_policy(root / "profiles" / "beta", 333)

    cases = [
        # per-profile home (standalone gateway per profile)
        (root / "profiles" / "alpha", "alpha", 222),
        (root / "profiles" / "alpha", "beta", 333),
        (root / "profiles" / "alpha", None, 222),
        (root / "profiles" / "beta", "beta", 333),
        # root home (multiplexed gateway)
        (root, "alpha", 222),
        (root, "beta", 333),
        (root, "default", 111),
        (root, None, 111),
        # "default" names the root profile's config, even from a profile home
        (root / "profiles" / "alpha", "default", 111),
    ]
    for home, profile, minutes in cases:
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
        monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
        policy = plugin._read_session_reset_policy(profile)
        assert policy == {"mode": "idle", "idle_minutes": minutes}, (
            f"HERMES_HOME={home} profile={profile!r} -> {policy}"
        )


# ---------------------------------------------------------------------------
# Event-loop hygiene for the clock stamp
#
# Alert sig:system|bacda61e84 / sig:system|e001fd8d32: the gateway asyncio loop
# blocked past Discord's 10 s heartbeat deadline because this async hook ran the
# within-threshold clock save inline. set_session_metadata -> _update_entry ->
# _save() rewrote the whole routing index in SQLite while the engine held the
# store lock across it. The write must run off the loop, take the engine's
# single-row identity-checked seam, and never land on a replacement route.
# ---------------------------------------------------------------------------

class _LockedStore:
    """Models the engine's store: ``set_session_metadata`` holds the lock across
    the single-row write; ``_update_entry`` is the older seam whose save
    rewrites the whole index; accessors need the same lock."""

    def __init__(self, entry):
        self._entry = entry
        self._lock = threading.Lock()
        self.stamp_delay = 0.4
        self.stamps = []
        self.used_update_entry = False
        self.used_setter = False

    def lookup_by_session_key(self, key):
        with self._lock:
            return self._entry

    def peek_session_id(self, key):
        with self._lock:
            return self._entry.session_id

    def set_session_metadata(self, key, meta_key, value, expected_session_id=None):
        self.used_setter = True
        with self._lock:
            time.sleep(self.stamp_delay)  # the synchronous row write
            if expected_session_id is not None and \
                    self._entry.session_id != expected_session_id:
                return False
            self._entry.metadata[meta_key] = value
            self.stamps.append(dict(self._entry.metadata))
        return True

    def _update_entry(self, key, mutate):
        self.used_update_entry = True
        with self._lock:
            time.sleep(self.stamp_delay)  # the synchronous full-index save
            if mutate(self._entry) is False:
                return False
            self.stamps.append(dict(self._entry.metadata))
            return True


def _entry_within_threshold():
    return SimpleNamespace(
        metadata={},
        updated_at=datetime.now(),
        created_at=datetime.now(),
        active_turn_token=None,
        session_id="seed-sid",
    )


def _replacement_entry():
    return SimpleNamespace(
        metadata={},
        updated_at=datetime.now(),
        created_at=datetime.now(),
        active_turn_token=None,
        session_id="rotated-sid",
    )


def _gateway():
    return SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:test-chat")


def _scenario(plugin, store):
    """Run the hook under a 10 ms ticker and count loop ticks it lets through."""
    event = SimpleNamespace(source=SimpleNamespace(profile=None), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}

    async def run():
        ticks = 0
        done = False

        async def ticker():
            nonlocal ticks
            while not done:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            await asyncio.sleep(0.03)
            base = ticks
            with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
                await plugin._maybe_reset(event, _gateway(), store)
            return ticks - base
        finally:
            done = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return asyncio.run(run())


def test_clock_stamp_does_not_block_the_event_loop(plugin):
    store = _LockedStore(_entry_within_threshold())

    ticks = _scenario(plugin, store)

    assert store.stamps, "the clock stamp must still be persisted"
    assert ticks >= 10, (
        f"only {ticks} loop ticks during a ~0.4 s save; the clock stamp blocked "
        "the event loop and can miss the Discord heartbeat"
    )


def test_clock_stamp_uses_the_single_row_setter(plugin):
    """The stamp must persist one row, not rewrite the whole routing index."""
    store = _LockedStore(_entry_within_threshold())

    _scenario(plugin, store)

    assert store.used_setter is True, (
        "the stamp did not use set_session_metadata, the engine's single-row write"
    )
    assert store.used_update_entry is False, (
        "the stamp fell back to _update_entry, whose save rewrites the whole "
        "routing index and blocks the gateway heartbeat"
    )
    assert store.stamps, "the clock stamp must still be persisted"


class _WholeIndexStore:
    """An engine older than the single-row metadata seam."""

    def __init__(self, entry):
        self._entry = entry
        self._lock = threading.Lock()
        self.stamps = []

    def lookup_by_session_key(self, key):
        with self._lock:
            return self._entry

    def _update_entry(self, key, mutate):
        with self._lock:
            if mutate(self._entry) is False:
                return False
            self.stamps.append(dict(self._entry.metadata))
            return True


def test_stamp_falls_back_to_update_entry_on_an_older_engine(plugin):
    store = _WholeIndexStore(_entry_within_threshold())

    assert plugin._stamp_clock(store, "route-1", "seed-sid", "t1") is True

    assert store.stamps and store.stamps[0][plugin._META_KEY] == "t1"


class _UnguardedSetterStore:
    """A store with a public metadata setter that cannot check the route.

    ``set_session_metadata(key, meta_key, value)`` names the route by its key
    only: nothing makes the identity check and the write one step, so a route
    that rotated first would take the stamp.
    """

    def __init__(self, entry):
        self._entry = entry
        self.writes = []

    def lookup_by_session_key(self, key):
        return self._entry

    def set_session_metadata(self, key, meta_key, value):
        self._entry.metadata[meta_key] = value
        self.writes.append((key, meta_key, value))
        return True


def test_a_setter_without_the_route_guard_never_stamps_a_replacement(plugin):
    """Finding (PR #2, __init__.py:319): the route already rotated, the stamp
    still carries the old session id, and the store's setter has no way to
    refuse it. The durable write is skipped: the old conversation's activity
    must not land on the replacement and move its idle clock."""
    store = _UnguardedSetterStore(_replacement_entry())  # route rotated first

    key = "route-rotated"
    assert plugin._stamp_clock(store, key, "seed-sid", "2026-11-21T07:31:18") is False

    assert store.writes == [], "the old route's stamp was written through an unguarded setter"
    assert store._entry.metadata == {}, "the replacement route's clock was altered"
    # The process-local clock still advances, bound to the id it saw, so the
    # hook's idle decision for the live conversation is unharmed.
    assert plugin._local_clock(key, "seed-sid") is not None
    assert plugin._local_clock(key, "rotated-sid") is None


def test_a_setter_without_the_route_guard_keeps_the_local_clock_for_a_live_route(plugin):
    """Skipping the durable write must not stall the user-activity clock."""
    store = _UnguardedSetterStore(_entry_within_threshold())
    key = "route-live"

    assert plugin._stamp_clock(store, key, "seed-sid", "2026-11-21T07:31:18") is False

    assert store.writes == []
    read = plugin._last_user_inbound(store._entry, key)
    assert read is not None and read.isoformat() == "2026-11-21T07:31:18"


class _RotatesDuringWrite(_LockedStore):
    """The route is replaced just before the stamp takes the store lock."""

    def set_session_metadata(self, key, meta_key, value, expected_session_id=None):
        self.used_setter = True
        with self._lock:
            self._entry = _replacement_entry()
            time.sleep(self.stamp_delay)
            if expected_session_id is not None and \
                    self._entry.session_id != expected_session_id:
                return False
            self._entry.metadata[meta_key] = value
            self.stamps.append(dict(self._entry.metadata))
            return True


def test_clock_stamp_never_lands_on_the_replacement_route(plugin):
    """A reset between the read and the write wins.

    The engine refuses a stamp for a route it no longer names (it captures the
    id it saw), and this process's clock is bound to that id, so a replacement
    never inherits the old conversation's activity. The store double here has no
    such guard, which is exactly the point: the plugin must not read the
    replacement's stamp as its own clock.
    """
    store = _RotatesDuringWrite(_entry_within_threshold())

    _scenario(plugin, store)

    key = _gateway()._session_key_for_source(None)
    assert store.stamps == [], (
        "the clock landed on the replacement route instead of being refused"
    )
    assert plugin._local_clock(key, "rotated-sid") is None, (
        "the plugin treated the replacement route as stamped by the old one"
    )
    assert plugin._local_clock(key, "seed-sid") is not None


def test_a_refreshed_route_is_not_evicted_by_the_cap(plugin):
    """Eviction takes the least recently stamped route, not the oldest key."""
    saved = dict(plugin._LOCAL_CLOCKS)
    old_max = plugin._LOCAL_CLOCK_MAX
    try:
        plugin._LOCAL_CLOCKS.clear()
        plugin._LOCAL_CLOCK_MAX = 4  # capacity pressure evicts cap // 4 == one entry
        plugin._remember_clock("busy", "sid", "2026-11-21T10:50:00")
        plugin._remember_clock("a", "sid", "2026-11-21T10:00:00")
        plugin._remember_clock("b", "sid", "2026-11-21T10:00:00")
        plugin._remember_clock("busy", "sid", "2026-11-21T10:55:00")  # refreshed
        plugin._remember_clock("c", "sid", "2026-11-21T10:00:00")
        plugin._remember_clock("d", "sid", "2026-11-21T10:00:00")  # pushes past the cap

        assert plugin._local_clock("busy", "sid") is not None, (
            "the busiest route lost its only fresh clock to capacity pressure"
        )
        assert plugin._local_clock("a", "sid") is None, "nothing was evicted"
    finally:
        plugin._LOCAL_CLOCKS.clear()
        plugin._LOCAL_CLOCKS.update(saved)
        plugin._LOCAL_CLOCK_MAX = old_max


def test_a_declined_store_write_still_advances_the_clock(plugin):
    """A degraded engine returns False; the clock must not stall.

    Without this the next message reads a stale clock and the idle policy
    resets a conversation the user is in.
    """

    class _Declines(_LockedStore):
        def set_session_metadata(self, key, meta_key, value, expected_session_id=None):
            self.used_setter = True
            return False

    store = _Declines(_entry_within_threshold())
    entry = store._entry
    now = datetime.now()

    key = _gateway()._session_key_for_source(None)
    assert plugin._stamp_clock(store, key, entry.session_id, now.isoformat()) is False

    read = plugin._last_user_inbound(entry, key)
    assert read is not None and read.isoformat() == now.isoformat(), (
        "a declined store write stalled the user-activity clock"
    )



class _QueueGapStore(_LockedStore):
    """Models the executor queue gap: the stamp is submitted but nothing it does
    is visible yet, so an inline read still sees the old clock.

    ``queued``/``release`` are the gate; ``_gated_clock_publication`` places it
    BEFORE the plugin publishes its process-local clock. A gate inside the store
    call opens too late: the plugin has already advanced its own copy, a second
    message reads that copy, and the ordering tests passed with no route lock.
    """

    def __init__(self, entry):
        super().__init__(entry)
        self.stamp_delay = 0
        self.queued = threading.Event()
        self.release = threading.Event()

    def set_session_metadata(self, key, meta_key, value, expected_session_id=None):
        self.used_setter = True
        with self._lock:
            if expected_session_id is not None and \
                    self._entry.session_id != expected_session_id:
                return False
            self._entry.metadata[meta_key] = value
            self.stamps.append(dict(self._entry.metadata))
            return True


class _FakeDateTime(datetime):
    current = None

    @classmethod
    def now(cls, tz=None):
        return cls.current


@contextlib.contextmanager
def _gated_clock_publication(plugin, store):
    """Hold the stamp worker before it publishes the clock anywhere.

    The worker's first act is ``_remember_clock``; parking it there means no
    reader, local or durable, can see message 1's arrival until ``release``.
    """
    real = plugin._remember_clock

    def gated(session_key, session_id, value):
        store.queued.set()
        store.release.wait(timeout=5)  # pending: nothing published yet
        real(session_key, session_id, value)

    with mock.patch.object(plugin, "_remember_clock", gated):
        yield


def _run_overlap(plugin, *, cancel_first, route_lock=True):
    """Two messages on one route while the first stamp is pending.

    The first message's stamp is held open; the boundary is crossed; the second
    message arrives. With the route lock the second hook waits for the first
    stamp, reads the fresh clock and does not reset. Without the lock (a control,
    ``route_lock=False``) it reads the old clock and resets: the scenario can
    therefore fail, which is what makes the passing runs mean something.
    """
    # Patching the module's ``datetime`` shadows the isinstance() base class, so
    # every timestamp this scenario produces is a _FakeDateTime.
    base = _FakeDateTime(2026, 11, 21, 12, 0, 0)
    entry = SimpleNamespace(metadata={}, updated_at=base, created_at=base,
                            active_turn_token=None, session_id="seed-sid")
    store = _QueueGapStore(entry)
    funnel_calls = []

    def funnel(_event):
        funnel_calls.append(1)
        return None

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: "route-1",
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=SimpleNamespace(profile=None), internal=False)
    policy = {"mode": "idle", "idle_minutes": 1}  # 60 s boundary
    seen = SimpleNamespace(second_done_while_pending=None)

    original_datetime = plugin.datetime
    plugin.datetime = _FakeDateTime
    _FakeDateTime.current = _FakeDateTime(2026, 11, 21, 12, 0, 30)
    try:
        async def run():
            with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
                first = asyncio.create_task(plugin._maybe_reset(event, gateway, store))
                await asyncio.to_thread(store.queued.wait, 5)  # first stamp is pending
                if cancel_first:
                    first.cancel()
                    await asyncio.sleep(0)  # deliver the cancellation
                _FakeDateTime.current = _FakeDateTime(2026, 11, 21, 12, 1, 1)  # boundary crossed
                second = asyncio.create_task(plugin._maybe_reset(event, gateway, store))
                await asyncio.sleep(0.05)
                seen.second_done_while_pending = second.done()
                store.release.set()
                if cancel_first:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(first, timeout=10)
                    await asyncio.wait_for(second, timeout=10)
                else:
                    await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

        with _gated_clock_publication(plugin, store):
            if route_lock:
                asyncio.run(run())
            else:
                with mock.patch.object(plugin, "_route_lock_for",
                                       lambda _key: contextlib.nullcontext()):
                    asyncio.run(run())
    finally:
        plugin.datetime = original_datetime
    return SimpleNamespace(funnel_calls=funnel_calls, store=store, seen=seen)


def test_overlapping_messages_on_one_route_keep_their_order(plugin):
    """BLOCKER (round 6): message 2 must not read the clock before message 1's
    off-loop stamp lands, or a boundary crossed between them fires a reset that
    the baseline never fired."""
    out = _run_overlap(plugin, cancel_first=False)

    assert out.seen.second_done_while_pending is False, (
        "the second message got through the hook while the first stamp was pending"
    )
    assert out.funnel_calls == [], (
        "the second message read a stale clock and reset a session the first "
        "message had just stamped"
    )
    assert len(out.store.stamps) == 2, "both arrivals must still advance the clock"


def test_cancelled_stamp_still_serializes_the_route(plugin):
    """BLOCKER (round 7): cancelling the hook must not release the route lock
    while the save is still in flight, or the next message reads a stale clock."""
    out = _run_overlap(plugin, cancel_first=True)

    assert out.seen.second_done_while_pending is False, (
        "the second message got through the hook while the cancelled stamp was pending"
    )
    assert out.funnel_calls == [], (
        "a cancelled hook released the route lock mid-save, so the next message "
        "read a stale clock and reset the session"
    )
    assert out.store.stamps and len(out.store.stamps) == 2, \
        "both arrivals must still advance the clock"


def test_the_overlap_scenario_notices_a_missing_route_lock(plugin):
    """Negative control (RULES 13): without the route lock the second message
    resets the session, so the ordering test above can really fail. It once could
    not: the clock was published locally before the test's gate opened."""
    out = _run_overlap(plugin, cancel_first=False, route_lock=False)

    assert out.funnel_calls == [1], "the scenario cannot tell a missing route lock apart"


def test_the_cancel_scenario_notices_a_missing_route_lock(plugin):
    out = _run_overlap(plugin, cancel_first=True, route_lock=False)

    assert out.funnel_calls == [1], "the scenario cannot tell a missing route lock apart"


class _FailingSaveStore(_QueueGapStore):
    """The queued save raises after the hook is cancelled."""

    def set_session_metadata(self, key, meta_key, value, expected_session_id=None):
        self.queued.set()
        self.release.wait(timeout=5)
        raise OSError("state.db is gone")


def test_failed_save_during_cancellation_keeps_the_cancellation(plugin):
    """BLOCKER (round 8): a save error must not replace CancelledError, or the
    gateway treats it as a plugin failure and keeps dispatching."""
    base = _FakeDateTime(2026, 11, 21, 12, 0, 0)
    entry = SimpleNamespace(metadata={}, updated_at=base, created_at=base,
                            active_turn_token=None, session_id="seed-sid")
    store = _FailingSaveStore(entry)

    gateway = SimpleNamespace(_session_key_for_source=lambda src: "route-1")
    event = SimpleNamespace(source=SimpleNamespace(profile=None), internal=False)
    policy = {"mode": "idle", "idle_minutes": 1}

    original_datetime = plugin.datetime
    plugin.datetime = _FakeDateTime
    _FakeDateTime.current = _FakeDateTime(2026, 11, 21, 12, 0, 30)
    try:
        async def run():
            with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
                hook = asyncio.create_task(plugin._maybe_reset(event, gateway, store))
                await asyncio.to_thread(store.queued.wait, 5)
                hook.cancel()
                await asyncio.sleep(0)
                store.release.set()
                await asyncio.wait_for(hook, timeout=10)

        raised = None
        try:
            asyncio.run(run())
        except BaseException as exc:  # noqa: BLE001 - we assert on the type below
            raised = exc
        assert isinstance(raised, asyncio.CancelledError), (
            f"expected CancelledError, got {raised!r}; a failed save must not "
            "replace the cancellation"
        )
    finally:
        plugin.datetime = original_datetime


# ---------------------------------------------------------------------------
# Two routes, one store: a stamp on one route must not hold up another's lookup
#
# Finding (PR #1, __init__.py:247): moving the save to a worker is not enough
# while the engine's stamp is a whole-index save under the store lock: the next
# route's inline lookup (it runs on the gateway loop) waits for that lock. The
# engine's identity-checked single-row setter never takes the whole-index path.
# This test runs the real plugin over a real SessionStore, keeps the whole-index
# writer busy behind a gate, and times the second route.
# ---------------------------------------------------------------------------

# Set to 1 only for an install made BEFORE the engine's single-row setter is
# live (job oc-fix-engine-recarry-20261122): the test then reports xfail with the
# reason instead of failing. Unset, an engine without the setter fails the suite.
ENGINE_FASTPATH_PENDING_ENV = "SRP_ENGINE_FASTPATH_PENDING"


def _engine_has_one_row_setter() -> bool:
    mod = _import_tree_module("gateway.session")
    return "expected_session_id" in inspect.signature(
        mod.SessionStore.set_session_metadata).parameters


def _xfail_while_the_engine_carry_is_pending() -> None:
    """Skip when there is no tree; xfail only on the explicit pending override."""
    if not _engine_has_one_row_setter() and os.environ.get(ENGINE_FASTPATH_PENDING_ENV) == "1":
        pytest.xfail("engine has no identity-checked single-row setter yet "
                     "(carried by job oc-fix-engine-recarry-20261122)")


def _two_route_contention(plugin, tmp_path, monkeypatch, *, row_write_under_store_lock=False):
    """Route A's stamp is parked INSIDE the engine's database write; route B's
    hook runs meanwhile, under a ticker that measures the gateway loop's pauses.

    The gate sits at the database boundary (``SessionDB._execute_write``), below
    whatever lock the engine holds, so it models slow SQLite I/O wherever the
    engine does it: a whole-index save under the store lock stalls route B's
    inline lookup, a single-row write outside that lock does not. ``entered`` is
    set by the writer itself, so route B only runs once A is really inside it.
    ``row_write_under_store_lock`` is the negative control: the single-row write
    is forced to run under the store lock, as a bad engine would do it.
    """
    store = _make_real_store(tmp_path)
    key_a, _entry_a = _seed_route(store, chat_id="route-a")
    key_b, _entry_b = _seed_route(store, chat_id="route-b")

    busy_for = 1.0                  # the longest the parked write stays parked
    entered = threading.Event()     # set by the writer once route A is inside it
    release = threading.Event()     # the gate: the writer finishes when this opens
    claim = threading.Lock()
    armed = threading.Event()
    expired = []
    whole_index_saves = []

    def park():
        with claim:
            mine = armed.is_set()
            armed.clear()           # only the first write after arming is parked
        if mine:
            entered.set()
            if not release.wait(timeout=busy_for):
                expired.append(1)   # somebody waited out the whole gate

    # The database boundary, where an engine with a SQLite routing table writes.
    session_db = _import_tree_module("hermes_state").SessionDB
    real_write = getattr(session_db, "_execute_write", None)
    if real_write is not None:
        def parked_write(db, *args, **kwargs):
            park()
            return real_write(db, *args, **kwargs)

        monkeypatch.setattr(session_db, "_execute_write", parked_write)

    # The whole-index save, parked too: an engine without a SQLite routing table
    # (or one that falls back to the full rewrite) writes there, under its lock.
    real_save = store._save

    def counted_save(*args, **kwargs):
        whole_index_saves.append(1)
        park()
        return real_save(*args, **kwargs)

    store._save = counted_save
    if row_write_under_store_lock:
        real_row = store._save_entry

        def row_under_lock(*args, **kwargs):
            with store._lock:
                return real_row(*args, lock_held=True, **kwargs)

        store._save_entry = row_under_lock

    policy = {"mode": "idle", "idle_minutes": 60}  # both routes are fresh: stamp path
    events = {key: SimpleNamespace(source=SimpleNamespace(profile=None), internal=False)
              for key in (key_a, key_b)}

    def hook(key):
        gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
        return plugin._maybe_reset(events[key], gateway, store)

    async def run():
        ticks = []
        running = True

        async def ticker():
            while running:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
            armed.set()
            first = asyncio.create_task(hook(key_a))
            got_in = await asyncio.to_thread(entered.wait, 5)  # A is inside the write
            watcher = asyncio.create_task(ticker())
            await asyncio.sleep(0.05)
            second = asyncio.create_task(hook(key_b))  # its lookup runs inline, on the loop
            await asyncio.sleep(0.4)                   # the window with A parked
            running = False
            await watcher
            release.set()
            await second
            await first
            # The gateway loop's longest pause. Route B's OWN stamp may wait for the
            # engine's write lock off the loop; only a stalled loop misses a heartbeat.
            return got_in, max(b - a for a, b in zip(ticks, ticks[1:]))

    got_in, loop_gap = asyncio.run(run())
    return SimpleNamespace(store=store, key_a=key_a, key_b=key_b, got_in=got_in,
                           loop_gap=loop_gap, expired=expired, busy_for=busy_for,
                           whole_index_saves=whole_index_saves)


def test_a_stamp_on_one_route_never_holds_up_another_routes_lookup(plugin, tmp_path, monkeypatch):
    _xfail_while_the_engine_carry_is_pending()

    out = _two_route_contention(plugin, tmp_path, monkeypatch)

    assert out.got_in, "route A's stamp never reached the database write"
    assert out.loop_gap < out.busy_for / 4 and not out.expired, (
        f"the gateway loop paused {out.loop_gap:.3f} s behind route A's stamp: route "
        "B's inline lookup waited for the store lock the stamp holds. The engine must "
        "carry the identity-checked single-row setter (job oc-fix-engine-recarry-20261122)"
    )
    assert out.whole_index_saves == [], (
        "the stamp rewrote the whole routing index; the engine has no "
        "identity-checked single-row setter for the plugin to use"
    )
    assert out.store.get_session_metadata(out.key_a, plugin._META_KEY) is not None
    assert out.store.get_session_metadata(out.key_b, plugin._META_KEY) is not None


def test_the_contention_test_notices_a_row_write_under_the_store_lock(plugin, tmp_path,
                                                                      monkeypatch):
    """Negative control: an engine that runs its single-row write under the store
    lock stalls the loop, and the contention test above can see it. Needs the
    engine's single-row path, whose write this control wraps."""
    _xfail_while_the_engine_carry_is_pending()

    out = _two_route_contention(plugin, tmp_path, monkeypatch, row_write_under_store_lock=True)

    assert out.got_in
    assert out.loop_gap >= out.busy_for / 4 or out.expired, (
        f"the loop was not held up ({out.loop_gap:.3f} s) although the row write held "
        "the store lock: the contention test cannot tell a lock-holding write apart"
    )


# ---------------------------------------------------------------------------
# stamp-probe.py: the real-flow probe of the stamp, and its negative control
#
# Finding (probe): its rotation proof read the local cache and a write count, so
# a plugin that dropped the route guard still printed PROBE GREEN. The probe must
# now see the stale stamp refused and find the replacement's SQLite row clean;
# this test proves it by running the probe against a plugin that lost the guard.
# ---------------------------------------------------------------------------

PROBE = PLUGIN_DIR / "stamp-probe.py"


def _run_probe(plugin_init: Path, cwd: Path) -> "subprocess.CompletedProcess[str]":
    env = dict(os.environ, CL_TREE=str(HERMES_TREE), CL_PLUGIN=str(plugin_init),
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", str(PROBE)], capture_output=True,
                          text=True, timeout=120, env=env, cwd=cwd)


def test_the_stamp_probe_is_green_on_the_real_plugin(tmp_path):
    _xfail_while_the_engine_carry_is_pending()

    out = _run_probe(PLUGIN_INIT, tmp_path)

    assert out.returncode == 0 and "PROBE GREEN" in out.stdout, out.stdout + out.stderr


def test_the_stamp_probe_goes_red_when_the_route_guard_is_dropped(tmp_path):
    """Negative control: the same plugin minus ``expected_session_id`` stamps the
    replacement route, and the probe must say so."""
    _xfail_while_the_engine_carry_is_pending()
    source = PLUGIN_INIT.read_text()
    guarded_call = ("value,\n                               "
                    "expected_session_id=expected_session_id))")
    assert guarded_call in source, "the guarded setter call moved: update this control"
    mutant = tmp_path / "mutant_init.py"
    mutant.write_text(source.replace(guarded_call, "value))", 1))

    out = _run_probe(mutant, tmp_path)

    assert out.returncode != 0 and "PROBE RED" in out.stdout, (
        "the probe stayed green for a plugin that stamps the replacement route:\n"
        + out.stdout + out.stderr
    )
