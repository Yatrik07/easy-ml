import contextvars
import math
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from flask import request
from flask_socketio import SocketIO, join_room

from easyml.utils.cancellation import current_job, get_job

# Will be initialized in app.py
socketio = None

# Receives progress entries emitted outside a background job (e.g. `easy-ml retrain` in a terminal).
_progress_listener = contextvars.ContextVar("easyml_progress_listener", default=None)


@contextmanager
def progress_listener(callback):
    """Call ``callback(entry)`` for every progress update emitted outside a job in this context."""
    token = _progress_listener.set(callback)
    try:
        yield
    finally:
        _progress_listener.reset(token)

def init_socketio(app, allowed_origins=None):
    """
    Initializes the SocketIO object for real-time communication.

    Args:
        app (Flask): Flask application instance.
        allowed_origins (list or callable, optional): Origins allowed to connect, as accepted by
            python-engineio. By default only the page served by this server (same origin) may connect.

    Returns:
        SocketIO: Initialized SocketIO object.
    """

    global socketio
    socketio = SocketIO(app, cors_allowed_origins=allowed_origins or None, async_mode='threading',
                        ping_timeout=120, ping_interval=30)
    socketio.on_event("join_run", _join_run)
    app.wsgi_app = _EndWerkzeugWebSocketsCleanly(app.wsgi_app)
    return socketio


class _EndWerkzeugWebSocketsCleanly:
    """
    Stop the Werkzeug dev server from writing an HTTP response onto a closed WebSocket.

    In ``threading`` mode simple-websocket performs the WebSocket handshake directly on the raw
    socket, and the WSGI call only returns once the WebSocket has closed. Werkzeug then still
    writes an HTTP response down the upgraded socket: a ``500`` page ("write() before
    start_response") for a direct WebSocket connection, or engine.io's ``200 OK`` for a connection
    upgraded from long-polling. Browsers parse those bytes as a WebSocket frame and log
    "WebSocket connection ... failed: Invalid frame header". Reporting the finished WebSocket as a
    dropped connection makes Werkzeug end the request without writing anything. Error statuses
    (e.g. a rejected origin) are sent before any handshake, so they still reach the client.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        if "werkzeug.socket" not in environ or environ.get("HTTP_UPGRADE", "").lower() != "websocket":
            return self.wsgi_app(environ, start_response)

        statuses = []

        def tracking_start_response(status, headers, exc_info=None):
            statuses.append(status)
            return start_response(status, headers, exc_info)

        result = self.wsgi_app(environ, tracking_start_response)
        if not statuses or statuses[-1][:1] in ("1", "2"):
            if hasattr(result, "close"):
                result.close()
            raise ConnectionAbortedError("WebSocket closed")
        return result


def _join_run(data):
    """
    Subscribe the calling client to one run's progress and replay what it has missed.

    Holding the job lock while joining means every event is delivered exactly once: either it was
    recorded before the join (and is replayed here) or it is emitted to the room afterwards.
    Clients may still see a replayed event twice after a reconnect, so they dedupe on ``seq``.
    """
    job = get_job((data or {}).get("job_id"))
    if job is None:
        return {"ok": False, "error": "Unknown training job."}
    with job.lock:
        join_room(job.room)
        for entry in job.history:
            socketio.emit("progress", entry, to=request.sid)
    return {"ok": True, "finished": job.finished}


def _to_jsonable(obj):
    """Convert NumPy values and NaN/inf (invalid JSON for the browser) to plain Python values."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    return obj

def emit_progress(msg: str, step: int = None, total_steps: int = None, pct: int = None, final: bool = False, result: dict = None, epoch_data: dict = None):
    """
    Emits a progress update through WebSocket.

    Args:
        msg (str): Progress message.
        step (int): Current step number (1-based).
        total_steps (int): Total number of steps.
        pct (int): Explicit percentage override (0-100). If None, computed from step/total_steps.
        final (bool): Whether this is the final update.
        result (dict, optional): Final result payload.
    """
    # Compute percentage
    computed_pct = pct
    if computed_pct is None and step is not None and total_steps and total_steps > 0:
        computed_pct = int((step / total_steps) * 100)

    entry = {
        "message": msg,
        "timestamp": datetime.now().isoformat(),
        "done": final,
        "pct": computed_pct,
    }
    if result is not None:
        entry["result"] = _to_jsonable(result)

    if epoch_data is not None:
        entry["epoch_data"] = _to_jsonable(epoch_data)

    job = current_job()
    if job is not None:
        # Progress belongs to one run: only clients that joined its room receive it.
        with job.lock:
            job.record(entry)
            if socketio:
                socketio.emit("progress", entry, to=job.room)
        if final:
            job.finished = True
    else:
        listener = _progress_listener.get()
        if listener is not None:
            try:
                listener(entry)
            except Exception:
                pass  # a display problem must never stop training
        from easyml.utils.logger import write_log
        write_log(f"[Progress] {entry['message']}", "info", "socket_emitter.py")