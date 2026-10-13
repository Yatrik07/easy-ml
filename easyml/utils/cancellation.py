"""
Per-run training jobs: an id, a cancel flag and the progress history of each background run.

The job a thread is working for lives in a context variable, so the training code of every task
family can call :func:`check_cancelled` / :func:`is_cancelled` (and the socket emitter can find the
run's room) without passing the job around. Code running outside a job (e.g. ``MLTrainer.run()``
from a script) is never cancelled and emits no socket events.

Lives in a dependency-free module so tabular and NLP training never import torch.
"""

import contextvars
import threading
import uuid
from collections import OrderedDict

MAX_FINISHED_JOBS = 50
MAX_HISTORY = 5000

_current_job = contextvars.ContextVar("easyml_training_job", default=None)
_jobs = OrderedDict()
_jobs_lock = threading.Lock()


class TrainingJob:
    """A single background training run."""

    def __init__(self, label=""):
        self.job_id = uuid.uuid4().hex
        self.label = label
        self.cancel_event = threading.Event()
        self.finished = False
        # Guards ``history`` and ``seq`` so a client joining the room gets every event exactly once.
        self.lock = threading.Lock()
        self.history = []
        self.seq = 0

    @property
    def room(self):
        return f"run:{self.job_id}"

    def cancel(self):
        self.cancel_event.set()

    def record(self, entry):
        """Stamp ``entry`` with the next sequence number and keep it for clients that join late."""
        self.seq += 1
        entry["seq"] = self.seq
        entry["job_id"] = self.job_id
        self.history.append(entry)
        if len(self.history) > MAX_HISTORY:
            # Keep the first events (run setup) and the most recent ones.
            del self.history[100:len(self.history) - MAX_HISTORY + 100]
        return entry


def register_job(label=""):
    job = TrainingJob(label)
    with _jobs_lock:
        _jobs[job.job_id] = job
        finished = [job_id for job_id, j in _jobs.items() if j.finished]
        for job_id in finished[:max(0, len(finished) - MAX_FINISHED_JOBS)]:
            del _jobs[job_id]
    return job


def get_job(job_id):
    with _jobs_lock:
        return _jobs.get(job_id) if isinstance(job_id, str) else None


def active_jobs():
    with _jobs_lock:
        return [job for job in _jobs.values() if not job.finished]


def current_job():
    return _current_job.get()


def set_current_job(job):
    """Bind ``job`` to the calling thread's context; returns a token for :func:`reset_current_job`."""
    return _current_job.set(job)


def reset_current_job(token):
    _current_job.reset(token)


def is_cancelled():
    job = _current_job.get()
    return job is not None and job.cancel_event.is_set()


def check_cancelled():
    """Raise ``InterruptedError`` if the user asked to stop the run this thread is working for."""
    if is_cancelled():
        raise InterruptedError("Training cancelled by user")
