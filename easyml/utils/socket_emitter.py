from flask_socketio import SocketIO
from datetime import datetime

# Will be initialized in app.py
socketio = None

def init_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio

def emit_progress(msg: str, final: bool = False):
    entry = {
        "message": msg,
        "timestamp": datetime.now().isoformat(),
        "done": final
    }
    if socketio:
        socketio.emit("progress", entry)
    else:
        print(f"[Progress] {entry['timestamp']} - {entry['message']}")