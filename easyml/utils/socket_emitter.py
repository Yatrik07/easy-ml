from flask_socketio import SocketIO
from datetime import datetime

# Will be initialized in app.py
socketio = None

def init_socketio(app):
    """
    Initializes the SocketIO object for real-time communication.

    Args:
        app (Flask): Flask application instance.

    Returns:
        SocketIO: Initialized SocketIO object.
    """

    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                        ping_timeout=120, ping_interval=30)
    return socketio

def emit_progress(msg: str, step: int = None, total_steps: int = None, pct: int = None, final: bool = False, result: dict = None):
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
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        entry["result"] = convert_numpy(result)

    if socketio:
        socketio.emit("progress", entry)
    else:
        print(f"[Progress] {entry['timestamp']} - {entry['message']}")