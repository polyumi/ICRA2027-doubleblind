"""
Browser front-end for picking a checkpoint and bringing up the whole inference stack.

Serving a checkpoint by hand means getting four things right at once — ``POLICY``,
``SERVE_TACTILE``, ``send_tactile`` and ``n_audio_rows`` — and three of them fail *silently* when
wrong: the policy simply runs on a frame it was never trained on. This lists every checkpoint on
the box with the settings each one implies, then drives the existing tmux panes so the server and
the client always agree.

It runs on the machine that holds the checkpoints and the Docker daemon (polyumi-server), and is browsed
from anywhere on the network. Stdlib only, deliberately: it has to start under ``/usr/bin/python3``
with nothing installed.

Metadata is read from files beside the checkpoint, never by loading it — a scan of thirty
checkpoints would otherwise mean paging in tens of gigabytes:

- locally-trained runs carry ``.hydra/config.yaml``, which states ``sensor_group`` and
  ``audio_obs_horizon`` outright;
- cluster runs carry ``selection.json``, which names the policy and variant but *not* the audio
  horizon, so that one is inferred from the run date and flagged as such in the UI.

Usage::

    /usr/bin/python3 tools/policy_launcher.py --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import yaml

#: Where checkpoints live, relative to the repo root.
DATA_SUBDIR = 'data'

#: tmux targets created by fr3_session.sh. The launcher types into these rather than spawning its
#: own processes, so a run stays visible, killable, and survives this server dying.
SERVER_PANE = 'polyumi:0.0'
CLIENT_PANE = 'polyumi-ros:0.0'

#: The audio contract changed on this date: rows per mic_0 window went 17 -> 10. Used only when a
#: checkpoint carries no config stating its own horizon.
AUDIO_ALIGNMENT_DATE = '2026-09-07'
ROWS_BEFORE_ALIGNMENT = 17
ROWS_AFTER_ALIGNMENT = 10

#: Sensor group implied by a cluster run's ``model`` field, for display only. Launch settings do
#: not depend on it — see ``LaunchPlan``.
GROUP_BY_MODEL = {
    'vista_v': 'v',
    'vista_vt': 'vt',
    'vista_va': 'va',
    'vista_vta': 'vta',
    'vista': 'vta',
    'sparsh_x': 'vta',
}


@dataclass
class Checkpoint:
    """One checkpoint on disk, with everything needed to serve it."""

    path: str
    size_mb: float
    policy: str
    model: str
    sensor_group: Optional[str]
    audio_rows: Optional[int]
    audio_rows_known: bool
    epoch: Optional[int]
    loss: Optional[str]
    dataset: Optional[str]
    source: str
    warnings: list = field(default_factory=list)


def _read_yaml(path: Path) -> Optional[dict]:
    """Load a YAML file, or return None if it is missing or unreadable."""
    try:
        with path.open() as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None


def _read_json(path: Path) -> Optional[dict]:
    """Load a JSON file, or return None if it is missing or unreadable."""
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _find_hydra_config(ckpt: Path) -> Optional[dict]:
    """
    Find the ``.hydra/config.yaml`` for a locally-trained run.

    Checkpoints sit in ``<run>/checkpoints/<file>.ckpt`` while the config sits in
    ``<run>/.hydra/``, so walk up a few levels rather than assuming one depth.
    """
    for parent in list(ckpt.parents)[:4]:
        candidate = parent / '.hydra' / 'config.yaml'
        if candidate.is_file():
            return _read_yaml(candidate)
    return None


def _epoch_and_loss(name: str) -> tuple[Optional[int], Optional[str]]:
    """Pull the epoch number and whichever loss the filename records out of a checkpoint name."""
    epoch_match = re.search(r'epoch=(\d+)', name)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    loss_match = re.search(r'((?:val|train)_loss=[\d.]+)', name)
    return epoch, loss_match.group(1) if loss_match else None


def _from_hydra(cfg: dict) -> dict:
    """Extract the fields we care about from a Hydra config, tolerating missing keys."""
    task = cfg.get('task') or {}
    policy_cfg = cfg.get('policy') or {}
    target = str(policy_cfg.get('_target_', '') or cfg.get('_target_', ''))
    is_dp = 'diffusion_unet' in target or 'diffusion_policy' in target
    return {
        'policy': 'dp' if is_dp else 'vista',
        'model': str(cfg.get('exp_name') or cfg.get('name') or '?'),
        'sensor_group': policy_cfg.get('sensor_group'),
        'audio_rows': task.get('audio_obs_horizon'),
        'dataset': task.get('name'),
    }


def _infer_audio_rows(run_date: Optional[str]) -> int:
    """
    Guess the mic_0 window length from a run's date.

    Only used for cluster runs, whose ``selection.json`` does not record the horizon. Runs from
    the alignment date onward use the 10-row window; earlier ones use 17.
    """
    if run_date and run_date >= AUDIO_ALIGNMENT_DATE:
        return ROWS_AFTER_ALIGNMENT
    return ROWS_BEFORE_ALIGNMENT


def describe(ckpt: Path) -> Checkpoint:
    """
    Read everything known about one checkpoint from the files beside it.

    Never opens the checkpoint itself; see the module docstring for why.
    """
    warnings: list = []
    size_mb = round(ckpt.stat().st_size / 1048576, 1)
    epoch, loss = _epoch_and_loss(ckpt.name)

    cfg = _find_hydra_config(ckpt)
    if cfg:
        info = _from_hydra(cfg)
        rows = info['audio_rows']
        known = True
        if info['policy'] == 'dp':
            # dp is visuomotor: it has no mic_0 input, so there is no window to get wrong.
            rows = None
        elif rows is None:
            rows = ROWS_AFTER_ALIGNMENT
            known = False
            warnings.append('config has no audio_obs_horizon; assuming 10')
        return Checkpoint(
            path=str(ckpt),
            size_mb=size_mb,
            policy=info['policy'],
            model=info['model'],
            sensor_group=info['sensor_group'],
            audio_rows=None if rows is None else int(rows),
            audio_rows_known=known,
            epoch=epoch,
            loss=loss,
            dataset=info['dataset'],
            source='.hydra/config.yaml',
            warnings=warnings,
        )

    selection = _read_json(ckpt.parent / 'selection.json')
    if selection:
        model = str(selection.get('model') or selection.get('variant') or '?')
        policy = str(selection.get('policy') or 'vista')
        rows: Optional[int] = None
        rows_known = True
        if policy != 'dp':
            rows = _infer_audio_rows(selection.get('run_date'))
            rows_known = False
            warnings.append(f'audio rows inferred from run date {selection.get("run_date")}')
        return Checkpoint(
            path=str(ckpt),
            size_mb=size_mb,
            policy=policy,
            model=model,
            sensor_group=GROUP_BY_MODEL.get(model),
            audio_rows=rows,
            audio_rows_known=rows_known,
            epoch=selection.get('best_epoch', epoch),
            loss=loss,
            dataset=selection.get('dataset'),
            source='selection.json',
            warnings=warnings,
        )

    # Nothing beside it. Fall back to the path, which is the weakest signal we have.
    lowered = str(ckpt).lower()
    policy = 'dp' if '/dp' in lowered or 'dp_' in lowered else 'vista'
    warnings.append('no config or selection.json; settings guessed from the path')
    return Checkpoint(
        path=str(ckpt),
        size_mb=size_mb,
        policy=policy,
        model='?',
        sensor_group=None,
        audio_rows=None if policy == 'dp' else ROWS_AFTER_ALIGNMENT,
        audio_rows_known=policy == 'dp',
        epoch=epoch,
        loss=loss,
        dataset=None,
        source='path only',
        warnings=warnings,
    )


def scan(data_root: Path) -> list:
    """Describe every ``*.ckpt`` under ``data_root``, newest path first."""
    found = sorted(data_root.rglob('*.ckpt'))
    return [describe(p) for p in found]


@dataclass
class LaunchPlan:
    """The exact commands a checkpoint implies, ready to be typed into tmux."""

    serve: str
    launch: str
    policy: str
    serve_tactile: int
    send_tactile: bool
    audio_rows: int


def plan_for(
    ckpt: Checkpoint, repo: str, audio_rows: int, execute_motion: bool, pi_host: str, video_device: str, server_url: str
) -> LaunchPlan:
    """
    Build the serve and launch command lines for one checkpoint.

    Tactile is all-or-nothing on purpose. Every Vista ablation's normalizer carries parameters for
    ``finger_rgb`` and ``mic_0`` whether or not that variant consumes them, and ``encode_condition``
    reads only the channels its ``sensor_group`` declares — so sending everything is correct for
    ``v``, ``vt``, ``va`` and ``vta`` alike. Sending *less* is what breaks: a ``vt`` model indexes
    ``obs['finger_rgb']`` unconditionally. dp takes neither channel.
    """
    is_vista = ckpt.policy != 'dp'
    serve_tactile = 1 if is_vista else 0
    serve = (
        f'cd {shlex.quote(repo)} && POLICY={"vista" if is_vista else "dp"} '
        f'SERVE_TACTILE={serve_tactile} CKPT={shlex.quote(ckpt.path)} ./serve_policy.sh'
    )
    launch = (
        f'cd {shlex.quote(repo)} && source setup_franka_env.sh && '
        f'ros2 launch polyumi_ros2 inference_demo.launch.xml '
        f'inference_server_url:={server_url} '
        f'execute_motion:={"true" if execute_motion else "false"} '
        f'max_image_age_s:=0.3 '
        f'pi_host:={pi_host} video_device:={video_device} '
        f'send_tactile:={"true" if is_vista else "false"}'
    )
    if is_vista:
        launch += f' n_audio_rows:={audio_rows}'
    return LaunchPlan(
        serve=serve,
        launch=launch,
        policy=ckpt.policy,
        serve_tactile=serve_tactile,
        send_tactile=is_vista,
        audio_rows=audio_rows,
    )


def _coerce_rows(requested: Any, fallback: Optional[int]) -> Optional[int]:
    """
    Resolve the mic_0 window length, tolerating a checkpoint that has none.

    dp is visuomotor, so both the request and the checkpoint carry ``None`` and the launch line
    simply omits ``n_audio_rows``.
    """
    value = requested if requested is not None else fallback
    return None if value is None else int(value)


def tmux(*args: str) -> tuple[int, str]:
    """Run one tmux command, returning its exit status and combined output."""
    proc = subprocess.run(['tmux', *args], capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def pane_exists(pane: str) -> tuple[bool, str]:
    """
    Report whether a tmux target resolves, and what tmux said if it does not.

    ``list-panes`` rather than ``display-message``: the latter falls back to the current session
    for an unknown target and exits 0, so it answers "yes" for panes that do not exist.
    """
    status, output = tmux('list-panes', '-t', pane)
    return status == 0, output


def send_to_pane(pane: str, command: str) -> tuple[bool, str]:
    """
    Interrupt whatever a pane is running, then type a command into it and press Enter.

    Ctrl-C first because tmux ``send-keys`` appends to readline's buffer rather than replacing it;
    without it a second launch concatenates onto the first and submits the result.

    Returns tmux's verdict rather than discarding it: typing into a pane that does not exist fails
    with status 1 and no other symptom, so an unchecked send looks identical to a successful one
    until the wait for the server times out minutes later.
    """
    status, output = tmux('send-keys', '-t', pane, 'C-c')
    if status != 0:
        return False, output
    time.sleep(1.5)
    status, output = tmux('send-keys', '-t', pane, command, 'Enter')
    return status == 0, output


def stop_stack(log) -> None:
    """
    Halt the running stack: interrupt the client first, then the server.

    Client first on purpose. It is the only thing publishing target poses, so stopping it ends
    commanded motion; killing the server first would instead leave the client failing every POST
    while still running its control loop.

    This is not an emergency stop. The impedance controller on the NUC holds its last target when
    the commands stop, so the arm stays where it is under servo rather than going limp — use the
    hardware e-stop if the arm needs to be stopped for safety.
    """
    for pane, label in ((CLIENT_PANE, 'client'), (SERVER_PANE, 'server')):
        status, output = tmux('send-keys', '-t', pane, 'C-c')
        if status == 0:
            log(f'sent Ctrl-C to the {label} pane ({pane})')
        else:
            log(f'could not interrupt the {label} pane ({pane}): {output}')


def preflight(panes: tuple, pi_host: str, ports: tuple = (5555, 5556)) -> list:
    """
    Check the things this launcher assumes but does not create.

    It drives an ``fr3_session.sh`` layout; it does not build one. Returning the problems rather
    than raising lets the UI show all of them at once instead of one per attempt.
    """
    problems = []
    for pane in panes:
        ok, detail = pane_exists(pane)
        if not ok:
            problems.append(f'tmux pane {pane} not found ({detail}) — is fr3_session.sh up?')
    for port in ports:
        try:
            with socket.create_connection((pi_host, port), timeout=2):
                pass
        except OSError as exc:
            problems.append(
                f'Pi stream not answering at {pi_host}:{port} ({exc.__class__.__name__}) — '
                'is `polyumi-pi stream` running?'
            )
    return problems


def wait_for_server(url: str, log, cancelled=None, timeout_s: float = 420.0) -> bool:
    """
    Poll the policy server's docs endpoint until it answers, or give up.

    Reports progress while waiting, because the only legitimate reason this takes minutes is the
    image rebuilding — and a silent wait is indistinguishable from a wedged one.
    """
    deadline = time.time() + timeout_s
    probe = url.replace('/predict_cartesian/', '/docs')
    started = time.time()
    next_note = 30.0
    while time.time() < deadline:
        if cancelled is not None and cancelled():
            log('wait cancelled by stop')
            return False
        try:
            with urllib.request.urlopen(probe, timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        waited = time.time() - started
        if waited >= next_note:
            log(f'still waiting for the server ({int(waited)}s; image may be rebuilding)')
            next_note += 30.0
        time.sleep(2)
    return False


class LauncherState:
    """Shared state: the last scan, and a log of what the launcher has done."""

    def __init__(self, repo: Path):
        """Initialize with the repo root whose ``data/`` tree is scanned."""
        self.repo = repo
        self.lock = threading.Lock()
        self.checkpoints: list = []
        self.events: list = []
        self.busy = False
        # Set by /api/stop. The launch thread may be parked in wait_for_server for minutes, so it
        # has to be interruptible rather than only stoppable between steps.
        self.cancel = threading.Event()

    def log(self, message: str) -> None:
        """Append a timestamped line to the activity log the UI polls."""
        with self.lock:
            self.events.append(f'{time.strftime("%H:%M:%S")}  {message}')
            del self.events[:-200]

    def rescan(self) -> None:
        """Re-read the checkpoint tree from disk."""
        self.checkpoints = scan(self.repo / DATA_SUBDIR)
        self.log(f'scanned {len(self.checkpoints)} checkpoints')


PAGE = """<!doctype html>
<title>PolyUMI Policy Launcher</title>
<style>
 :root { color-scheme: light dark; --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; --acc:#0b5; }
 @media (prefers-color-scheme: dark) {
   :root { --bg:#111; --fg:#eee; --mut:#999; --line:#333; --acc:#3d8; } }
 body { margin:0; background:var(--bg); color:var(--fg);
        font:13px/1.5 ui-monospace,Menlo,Consolas,monospace; }
 header { padding:12px 16px; border-bottom:1px solid var(--line); }
 h1 { font-size:15px; margin:0 0 4px; }
 .mut { color:var(--mut); }
 main { display:grid; grid-template-columns:1fr 380px; gap:0; align-items:start; }
 @media (max-width:900px){ main { grid-template-columns:1fr; } }
 table { border-collapse:collapse; width:100%; }
 th,td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line);
         vertical-align:top; }
 th { position:sticky; top:0; background:var(--bg); font-weight:600; }
 tr:hover td { background:rgba(127,127,127,.10); cursor:pointer; }
 tr.sel td { background:rgba(0,187,85,.16); }
 .path { word-break:break-all; font-size:12px; }
 aside { padding:12px 16px; border-left:1px solid var(--line); position:sticky; top:0; }
 pre { background:rgba(127,127,127,.10); padding:8px; overflow-x:auto; white-space:pre-wrap;
       word-break:break-all; font-size:12px; }
 button { font:inherit; padding:7px 12px; cursor:pointer; }
 button.go { background:var(--acc); color:#000; border:0; font-weight:600; }
 button.stop { background:#c22; color:#fff; border:0; font-weight:700; padding:9px 18px; }
 .hdr { display:flex; justify-content:space-between; align-items:center; gap:16px; }
 button[disabled] { opacity:.5; cursor:not-allowed; }
 .warn { color:#c60; }
 label { display:block; margin:8px 0 2px; }
 #log { max-height:220px; overflow:auto; }
</style>
<header><div class="hdr">
  <div>
    <h1>PolyUMI Policy Launcher</h1>
    <div class="mut">Pick a checkpoint. Settings are read from its own config, then typed into the
      tmux panes. <b>execute_motion is on</b> &mdash; launching will move the arm.</div>
  </div>
  <button class="stop" onclick="stopAll()">STOP</button>
</div>
<div class="mut" style="margin-top:6px">STOP interrupts the client then the server. It is
  <b>not</b> an emergency stop &mdash; the impedance controller holds its last target. Use the
  hardware e-stop for safety.</div>
</header>
<main>
  <div><table id="tbl"><thead><tr>
    <th>dataset</th><th>model</th><th>policy</th><th>grp</th><th>rows</th>
    <th>epoch</th><th>loss</th><th>MB</th><th>path</th>
  </tr></thead><tbody></tbody></table></div>
  <aside>
    <button onclick="rescan()">Rescan</button>
    <div id="detail"><p class="mut">No checkpoint selected.</p></div>
    <div id="log"></div>
  </aside>
</main>
<script>
let rows = [], sel = null;
async function load() {
  rows = await (await fetch('api/checkpoints')).json();
  const tb = document.querySelector('#tbl tbody'); tb.innerHTML = '';
  rows.forEach((c, i) => {
    const tr = document.createElement('tr');
    tr.onclick = () => select(i);
    tr.innerHTML = `<td>${c.dataset||''}</td><td>${c.model}</td><td>${c.policy}</td>
      <td>${c.sensor_group||''}</td>
      <td>${c.audio_rows==null?'&mdash;':c.audio_rows}${c.audio_rows_known?'':'<span class="warn">?</span>'}</td>
      <td>${c.epoch??''}</td><td>${c.loss||''}</td><td>${c.size_mb}</td>
      <td class="path">${c.path}</td>`;
    tb.appendChild(tr);
  });
}
function select(i) {
  sel = i;
  document.querySelectorAll('#tbl tbody tr').forEach((t, j) =>
    t.classList.toggle('sel', j === i));
  const c = rows[i];
  const warn = c.warnings.length
    ? `<p class="warn">${c.warnings.map(w => '! ' + w).join('<br>')}</p>` : '';
  document.getElementById('detail').innerHTML = `
    <p class="path"><b>${c.path}</b></p>
    <p class="mut">settings from ${c.source}</p>${warn}
    ${c.audio_rows==null ? '<p class="mut">visuomotor: no mic_0 window</p>'
      : `<label>n_audio_rows (override if wrong)</label>
    <input id="rows" type="number" value="${c.audio_rows}" style="width:80px">`}
    <label><input id="motion" type="checkbox" checked> execute_motion</label>
    <p><button class="go" onclick="go()">Serve + launch</button></p>
    <pre id="plan"></pre>`;
  preview();
  const ri = document.getElementById('rows'); if (ri) ri.oninput = preview;
  document.getElementById('motion').onchange = preview;
}
async function preview() {
  const r = await fetch('api/plan', {method:'POST', body: JSON.stringify(body())});
  const p = await r.json();
  document.getElementById('plan').textContent =
    '# server pane\\n' + p.serve + '\\n\\n# client pane\\n' + p.launch;
}
function body() {
  return JSON.stringify ? {
    path: rows[sel].path,
    audio_rows: document.getElementById('rows')
      ? +document.getElementById('rows').value : null,
    execute_motion: document.getElementById('motion').checked
  } : {};
}
async function go() {
  if (!confirm('This starts the policy server and the ROS client.\\n'
      + (document.getElementById('motion').checked
         ? 'execute_motion is ON - the arm will move.' : 'Dry run.'))) return;
  document.querySelector('button.go').disabled = true;
  await fetch('api/launch', {method:'POST', body: JSON.stringify(body())});
  poll();
}
async function rescan() { await fetch('api/rescan', {method:'POST'}); load(); }
async function stopAll() {
  await fetch('api/stop', {method:'POST'});
  const b = document.querySelector('button.go'); if (b) b.disabled = false;
}
async function poll() {
  const ev = await (await fetch('api/events')).json();
  document.getElementById('log').innerHTML =
    '<pre>' + ev.events.slice(-25).join('\\n') + '</pre>';
  document.querySelector('button.go') &&
    (document.querySelector('button.go').disabled = ev.busy);
  setTimeout(poll, 1500);
}
load(); poll();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    """Serves the page and a small JSON API over it."""

    state: LauncherState
    repo: str
    pi_host: str
    video_device: str
    server_url: str

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default per-request stderr logging."""

    def _send(self, payload: Any, content_type: str = 'application/json') -> None:
        """Write one response body with the right headers."""
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        """Read and parse the request body as JSON."""
        length = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(length) or b'{}')

    def _selected(self, body: dict) -> Optional[Checkpoint]:
        """Find the scanned checkpoint the request names."""
        return next((c for c in self.state.checkpoints if c.path == body.get('path')), None)

    def _plan(self, body: dict) -> Optional[LaunchPlan]:
        """Build a launch plan from a request body, or None if the path is unknown."""
        ckpt = self._selected(body)
        if ckpt is None:
            return None
        return plan_for(
            ckpt,
            repo=self.repo,
            audio_rows=_coerce_rows(body.get('audio_rows'), ckpt.audio_rows),
            execute_motion=bool(body.get('execute_motion')),
            pi_host=self.pi_host,
            video_device=self.video_device,
            server_url=self.server_url,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's required name
        """Serve the page, the checkpoint list, or the activity log."""
        if self.path.startswith('/api/checkpoints'):
            self._send([asdict(c) for c in self.state.checkpoints])
        elif self.path.startswith('/api/events'):
            self._send({'events': self.state.events, 'busy': self.state.busy})
        else:
            self._send(PAGE.encode(), 'text/html; charset=utf-8')

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's required name
        """Preview a plan, rescan the tree, or run a launch."""
        if self.path.startswith('/api/stop'):
            self.state.cancel.set()
            self.state.log('STOP requested')
            stop_stack(self.state.log)
            self._send({'ok': True})
            return
        if self.path.startswith('/api/rescan'):
            self.state.rescan()
            self._send({'ok': True})
            return
        body = self._body()
        if self.path.startswith('/api/plan'):
            plan = self._plan(body)
            self._send(asdict(plan) if plan else {'error': 'unknown checkpoint'})
            return
        if self.path.startswith('/api/launch'):
            plan = self._plan(body)
            if plan is None:
                self._send({'error': 'unknown checkpoint'})
                return
            threading.Thread(target=self._run, args=(plan,), daemon=True).start()
            self._send({'ok': True})
            return
        self._send({'error': 'no such endpoint'})

    def _run(self, plan: LaunchPlan) -> None:
        """Drive the two tmux panes: serve, wait for readiness, then launch the client."""
        state = self.state
        state.cancel.clear()
        state.busy = True
        try:
            problems = preflight((SERVER_PANE, CLIENT_PANE), self.pi_host)
            if problems:
                for problem in problems:
                    state.log(f'ABORT: {problem}')
                state.log('nothing was typed into any pane.')
                return
            state.log(f'server pane <- POLICY={plan.policy} SERVE_TACTILE={plan.serve_tactile}')
            ok, detail = send_to_pane(SERVER_PANE, plan.serve)
            if not ok:
                state.log(f'ABORT: could not type into {SERVER_PANE} ({detail})')
                return
            state.log('waiting for the policy server to answer')
            if not wait_for_server(self.server_url, state.log, state.cancel.is_set):
                state.log('ERROR: server never answered; client NOT launched')
                return
            if state.cancel.is_set():
                state.log('stopped before the client was launched')
                return
            state.log('server is up')
            rows = 'n/a' if plan.audio_rows is None else plan.audio_rows
            state.log(f'client pane <- send_tactile={plan.send_tactile} n_audio_rows={rows}')
            ok, detail = send_to_pane(CLIENT_PANE, plan.launch)
            if not ok:
                state.log(f'ABORT: could not type into {CLIENT_PANE} ({detail}); the server is running with no client')
                return
            if plan.audio_rows is None:
                state.log('launched (visuomotor; no mic_0 window).')
            else:
                state.log(f'launched. Confirm the client banner says "mic_0 {rows} rows".')
        finally:
            state.busy = False


def main() -> None:
    """Parse arguments, scan once, and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(Path.home() / 'repos' / 'PolyUMI'))
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--pi-host', default='10.12.194.1')
    parser.add_argument('--video-device', default='/dev/video1')
    parser.add_argument('--server-url', default='http://localhost:8002/predict_cartesian/')
    args = parser.parse_args()

    state = LauncherState(Path(args.repo))
    state.rescan()

    Handler.state = state
    Handler.repo = args.repo
    Handler.pi_host = args.pi_host
    Handler.video_device = args.video_device
    Handler.server_url = args.server_url

    host = os.environ.get('LAUNCHER_BIND', '0.0.0.0')
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(
        f'policy launcher on http://{host}:{args.port}  '
        f'({len(state.checkpoints)} checkpoints under {args.repo}/{DATA_SUBDIR})'
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
