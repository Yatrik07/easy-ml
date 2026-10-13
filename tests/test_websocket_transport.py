"""The Werkzeug dev server must not write an HTTP response onto a closed WebSocket ("Invalid frame header")."""

import base64
import json
import os
import socket
import threading
import time
import urllib.request

import pytest
from werkzeug.serving import make_server


@pytest.fixture()
def server_port(app):
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port
    server.shutdown()


def _frame(opcode, payload=b""):
    mask = os.urandom(4)
    return bytes([0x80 | opcode, 0x80 | len(payload)]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _recv(sock, timeout=1.0):
    """Read whatever arrives within ``timeout`` or until the server closes the connection."""
    data, deadline = b"", time.time() + timeout
    while time.time() < deadline:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        if not chunk:
            break
        data += chunk
    return data


def _open_websocket(port, query):
    sock = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET /socket.io/?EIO=4&transport=websocket{query} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                  f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    assert _recv(sock).startswith(b"HTTP/1.1 101")
    return sock


def _close_like_a_browser(sock):
    # socket.io-client disconnect(): socket.io "41", then an empty WebSocket close frame.
    sock.sendall(_frame(1, b"41") + _frame(8))
    after_close = _recv(sock, timeout=3)
    sock.close()
    assert b"HTTP/1.1" not in after_close, after_close
    assert after_close == b"\x88\x00"  # nothing but the server's close frame


def test_direct_websocket_closes_cleanly(server_port):
    sock = _open_websocket(server_port, "")
    sock.sendall(_frame(1, b"40"))
    _close_like_a_browser(sock)


def test_websocket_upgraded_from_polling_closes_cleanly(server_port):
    # What browsers do: open a long-polling session, then upgrade it to a WebSocket.
    with urllib.request.urlopen(f"http://127.0.0.1:{server_port}/socket.io/?EIO=4&transport=polling") as response:
        sid = json.loads(response.read().decode()[1:])["sid"]
    sock = _open_websocket(server_port, f"&sid={sid}")
    sock.sendall(_frame(1, b"2probe"))
    assert b"3probe" in _recv(sock)
    sock.sendall(_frame(1, b"5"))  # upgrade
    _close_like_a_browser(sock)


def test_rejected_websocket_still_gets_its_error_response(server_port):
    sock = socket.create_connection(("127.0.0.1", server_port), timeout=0.2)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET /socket.io/?EIO=4&transport=websocket HTTP/1.1\r\nHost: 127.0.0.1:{server_port}\r\n"
                  f"Origin: https://evil.example\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    response = _recv(sock, timeout=3)
    sock.close()
    assert response.startswith(b"HTTP/1.1 400")
