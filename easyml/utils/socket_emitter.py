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
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio

def emit_progress(msg: str, final: bool = False):
    """
    Emits a progress update through WebSocket. If SocketIO is not initialized, logs to console.

    Args:
        msg (str): Progress message to be sent.
        final (bool): Indicates whether this is the final update in the process.
    """

    entry = {
        "message": msg,
        "timestamp": datetime.now().isoformat(),
        "done": final
    }
    if socketio:
        socketio.emit("progress", entry)
    else:
        print(f"[Progress] {entry['timestamp']} - {entry['message']}")