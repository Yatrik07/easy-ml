"""
Where schedules run: a background thread of the studio, or ``easy-ml schedule worker`` (headless).

Both call ``run_due`` every ``POLL_SECONDS``; the database claim makes it safe to run several at once.
The studio thread skips a pass while someone is training in the studio, so it never competes with them.
"""

import os
import socket
import threading
import time

from easyml.utils.logger import write_log

POLL_SECONDS = 30
FIRST_CHECK_SECONDS = 5
_studio = {"thread": None, "stop": None}


def scheduler_disabled():
    return os.environ.get("EASYML_SCHEDULER", "").strip().lower() in ("0", "false", "no", "off")


def studio_scheduler_active():
    thread = _studio["thread"]
    return bool(thread and thread.is_alive())


def start_studio_scheduler(socketio=None, poll_seconds=POLL_SECONDS):
    """Start the studio's scheduler thread (once). Returns the thread, or None when disabled."""
    if scheduler_disabled():
        return None
    if studio_scheduler_active():
        return _studio["thread"]
    from easyml.scheduling.engine import run_due
    from easyml.utils.cancellation import active_jobs

    stop = threading.Event()
    runner = f"studio:{socket.gethostname()}:{os.getpid()}"

    def notify(event):
        if socketio is not None:
            socketio.emit("schedule_event", {k: v for k, v in event.items() if not k.startswith("_")})

    def loop():
        wait = FIRST_CHECK_SECONDS
        while not stop.wait(wait):
            wait = poll_seconds
            if active_jobs():  # someone is training in the studio: check again later
                continue
            try:
                run_due(runner=runner, notify=notify)
            except Exception as exc:  # never let the thread die
                write_log(f"Scheduler pass failed: {exc}", "error", os.path.basename(__file__))

    thread = threading.Thread(target=loop, name="easyml-scheduler", daemon=True)
    _studio.update(thread=thread, stop=stop)
    thread.start()
    return thread


def stop_studio_scheduler():
    if _studio["stop"] is not None:
        _studio["stop"].set()
    _studio.update(thread=None, stop=None)


def worker(poll_seconds=POLL_SECONDS, out=None):
    """Run schedules until stopped (Ctrl+C). Returns an exit code."""
    import sys

    from easyml.scheduling import store
    from easyml.scheduling.engine import run_due

    out = out or sys.stdout

    def log(message):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=out, flush=True)

    runner = f"worker:{socket.gethostname()}:{os.getpid()}"
    count = len([row for row in store.list_schedules() if row.get("enabled")])
    log(f"Easy-ML scheduler started: {count} enabled schedule(s), checking every {poll_seconds} s. Press CTRL+C to stop.")
    try:
        while True:
            run_due(runner=runner, log=log)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        log("Scheduler stopped.")
        return 0
