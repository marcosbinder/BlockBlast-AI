"""
Fast ADB Socket Client & Coordinate Bounds Engine for Block Blast (Moto G54 5G).

Provides:
- High-speed direct TCP socket streaming to local ADB daemon (127.0.0.1:5037).
- In-memory PNG screencap capture and OpenCV decoding (<1.3s latency).
- Physical coordinate bounds validation (1080x2400) for Moto G54 5G.
- Safe finger release clamping (Y <= 1580) to strictly avoid the tray cancellation zone (Y >= 1600).
- Subprocess fallback when ADB TCP daemon is offline or unreachable.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Set up module logger
logger = logging.getLogger("adb_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Physical Display & Safe Clamping Constants (Motorola Moto G54 5G)
SCREEN_WIDTH: int = 1080
SCREEN_HEIGHT: int = 2400
SCREEN_DENSITY: int = 400

# Board & Tray Geometry Constants
BOARD_LEFT_X: float = 61.0
BOARD_RIGHT_X: float = 1018.0
BOARD_TOP_Y: float = 581.0
BOARD_BOTTOM_Y: float = 1537.0

# Tray cancel hazard boundary & safe release ceiling
TRAY_CANCEL_ZONE_START_Y: int = 1600
MAX_SAFE_RELEASE_Y: int = 1850
MIN_SAFE_RELEASE_Y: int = 580

# Default ADB connection parameters
DEFAULT_ADB_HOST: str = "127.0.0.1"
DEFAULT_ADB_PORT: int = 5037
DEFAULT_DEVICE_SERIAL: str = "ZF524K4RCM"
DEFAULT_CONNECT_TIMEOUT: float = 2.0
DEFAULT_SOCKET_TIMEOUT: float = 5.0

# Known ADB executable paths on host system
KNOWN_ADB_PATHS = [
    r"C:\Users\Marco\Documents\scrcpy-win64-v3.3.4\adb.exe",
    r"C:\platform-tools\adb.exe",
    r"C:\Program Files\Android\platform-tools\adb.exe",
    r"C:\Android\platform-tools\adb.exe",
]

PNG_MAGIC_HEADER: bytes = b"\x89PNG\r\n\x1a\n"


def find_adb_executable(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locate the ADB executable on the host system.
    Searches custom_path, standard installation locations, and system PATH.
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_adb = os.environ.get("ADB_PATH")
    if env_adb and os.path.isfile(env_adb):
        return env_adb

    for path in KNOWN_ADB_PATHS:
        if os.path.isfile(path):
            return path

    which_adb = shutil.which("adb")
    if which_adb and os.path.isfile(which_adb):
        return which_adb

    return None


def clean_png_bytes(raw_bytes: bytes) -> bytes:
    """
    Clean Windows CRLF corruption in raw PNG byte streams if present.
    Standard ADB 'exec:' emits pure binary, but shell wrappers may corrupt \n to \r\n.
    """
    if not raw_bytes:
        return raw_bytes

    # If stream already begins with valid PNG magic header, return as is
    if raw_bytes.startswith(PNG_MAGIC_HEADER):
        return raw_bytes

    # If stream has \r\r\n Windows ADB shell corruption, replace with \n
    if b"\r\r\n" in raw_bytes[:32]:
        cleaned = raw_bytes.replace(b"\r\r\n", b"\n")
        if cleaned.startswith(PNG_MAGIC_HEADER):
            return cleaned

    # If stream has \r\n replacement
    if b"\r\n" in raw_bytes[:32]:
        cleaned = raw_bytes.replace(b"\r\n", b"\n")
        if cleaned.startswith(PNG_MAGIC_HEADER):
            return cleaned

    # If magic header is offset (e.g. leading text/newlines), find it
    idx = raw_bytes.find(PNG_MAGIC_HEADER)
    if idx > 0:
        return raw_bytes[idx:]

    return raw_bytes


def clamp_coordinate_x(x: Union[int, float]) -> int:
    """Clamps X coordinate strictly within [0, SCREEN_WIDTH]."""
    return max(0, min(SCREEN_WIDTH, int(round(x))))


def clamp_coordinate_y(y: Union[int, float]) -> int:
    """Clamps Y coordinate strictly within [0, SCREEN_HEIGHT]."""
    return max(0, min(SCREEN_HEIGHT, int(round(y))))


def clamp_coordinates(x: Union[int, float], y: Union[int, float]) -> Tuple[int, int]:
    """Clamps (X, Y) coordinates strictly within physical screen bounds [0, 1080] x [0, 2400]."""
    return clamp_coordinate_x(x), clamp_coordinate_y(y)


def clamp_release_y(y: Union[int, float]) -> int:
    """
    Clamps finger release Y coordinate to Y <= MAX_SAFE_RELEASE_Y (1580px).
    Guarantees the touch gesture never releases in the tray cancellation zone (Y >= 1600px).
    """
    clamped_y = clamp_coordinate_y(y)
    return min(MAX_SAFE_RELEASE_Y, clamped_y)


def is_within_bounds(x: Union[int, float], y: Union[int, float]) -> bool:
    """Returns True if (x, y) is within [0, SCREEN_WIDTH] and [0, SCREEN_HEIGHT]."""
    return 0 <= x <= SCREEN_WIDTH and 0 <= y <= SCREEN_HEIGHT


def is_in_tray_cancel_zone(y: Union[int, float]) -> bool:
    """Returns True if Y coordinate falls into the tray cancellation hazard zone (Y >= 1600)."""
    return y >= TRAY_CANCEL_ZONE_START_Y


class FastADBSocketClient:
    """
    High-speed direct socket client communicating with the local ADB server daemon.
    Provides streaming raw screencap parsing, safe touch dispatching, and subprocess fallback.
    """

    def __init__(
        self,
        serial: str = DEFAULT_DEVICE_SERIAL,
        host: str = DEFAULT_ADB_HOST,
        port: int = DEFAULT_ADB_PORT,
        adb_path: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT,
    ) -> None:
        self.serial: str = serial
        self.host: str = host
        self.port: int = port
        self.connect_timeout: float = connect_timeout
        self.socket_timeout: float = socket_timeout
        self.adb_path: Optional[str] = find_adb_executable(adb_path)

        self._last_screencap_latency_ms: float = 0.0
        self._last_capture_method: str = "none"

        logger.debug(
            f"Initialized FastADBSocketClient (serial={self.serial}, host={self.host}:{self.port}, adb_path={self.adb_path})"
        )

    @property
    def last_screencap_latency_ms(self) -> float:
        """Returns the latency of the most recent screencap operation in milliseconds."""
        return self._last_screencap_latency_ms

    @property
    def last_capture_method(self) -> str:
        """Returns the capture method used for the most recent screencap ('socket' or 'subprocess')."""
        return self._last_capture_method

    def _create_socket(self) -> socket.socket:
        """Create and connect a TCP stream socket to the ADB server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.host, self.port))
            sock.settimeout(self.socket_timeout)
            return sock
        except Exception:
            sock.close()
            raise

    @staticmethod
    def _format_adb_request(payload: Union[str, bytes]) -> bytes:
        """Format a command string with 4-byte hex ASCII length prefix per ADB protocol."""
        if isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        else:
            payload_bytes = payload
        length_prefix = f"{len(payload_bytes):04x}".encode("ascii")
        return length_prefix + payload_bytes

    def _read_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Read exactly num_bytes from socket, raising EOFError if connection closes prematurely."""
        buf = bytearray()
        while len(buf) < num_bytes:
            chunk = sock.recv(num_bytes - len(buf))
            if not chunk:
                raise EOFError(f"Socket closed unexpectedly while reading {num_bytes} bytes (got {len(buf)} bytes)")
            buf.extend(chunk)
        return bytes(buf)

    def _read_status(self, sock: socket.socket) -> str:
        """Read 4-byte status from ADB server ('OKAY' or 'FAIL'). If FAIL, read error message."""
        status_bytes = self._read_exact(sock, 4)
        status_str = status_bytes.decode("ascii", errors="replace")
        if status_str == "FAIL":
            length_hex = self._read_exact(sock, 4).decode("ascii", errors="replace")
            try:
                err_len = int(length_hex, 16)
                err_msg = self._read_exact(sock, err_len).decode("utf-8", errors="replace")
            except Exception:
                err_msg = "Unknown error reading failure message"
            raise RuntimeError(f"ADB Server error: {err_msg}")
        return status_str

    def _switch_device_transport(self, sock: socket.socket) -> None:
        """Direct socket connection to target device serial via 'host:transport:<serial>'."""
        cmd = f"host:transport:{self.serial}"
        sock.sendall(self._format_adb_request(cmd))
        status = self._read_status(sock)
        if status != "OKAY":
            raise RuntimeError(f"ADB transport switch failed: status={status}")

    def execute_socket_command(self, service_cmd: str) -> bytes:
        """
        Execute an ADB service command (e.g. 'exec:screencap -p' or 'shell:input tap 100 100')
        over a fresh direct TCP socket connection and return the complete response payload.
        """
        sock = self._create_socket()
        try:
            self._switch_device_transport(sock)
            sock.sendall(self._format_adb_request(service_cmd))
            self._read_status(sock)

            # Read remaining response stream until EOF
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            sock.close()

    def _screencap_socket(self) -> np.ndarray:
        """Perform high-speed raw screencap over direct ADB TCP socket."""
        raw_bytes = self.execute_socket_command("exec:screencap -p")
        cleaned_bytes = clean_png_bytes(raw_bytes)
        if not cleaned_bytes:
            raise RuntimeError("Received empty screencap buffer from ADB socket")

        np_arr = np.frombuffer(cleaned_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed to parse {len(cleaned_bytes)} bytes as PNG image")

        return img

    def _screencap_subprocess(self) -> np.ndarray:
        """Fallback screencap using local adb.exe subprocess execution."""
        if not self.adb_path:
            raise RuntimeError("Cannot fallback to subprocess: adb executable not found on system")

        cmd = [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.socket_timeout + 3.0,
                check=True,
            )
            raw_bytes = clean_png_bytes(proc.stdout)
            if not raw_bytes:
                raise RuntimeError("Empty screencap payload received via subprocess")

            np_arr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                # Try fallback without exec-out (using shell)
                cmd_shell = [self.adb_path, "-s", self.serial, "shell", "screencap", "-p"]
                proc_shell = subprocess.run(
                    cmd_shell,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.socket_timeout + 3.0,
                    check=True,
                )
                raw_shell = clean_png_bytes(proc_shell.stdout)
                img = cv2.imdecode(np.frombuffer(raw_shell, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Failed to decode screencap buffer from subprocess")

            return img
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Subprocess screencap error: {e}") from e

    def screencap_cv2(self) -> np.ndarray:
        """
        Capture the current device screen and return as an OpenCV BGR numpy array (1080x2400).
        Automatically attempts direct socket capture first (~1.3s), falling back to subprocess if necessary.
        """
        t0 = time.perf_counter()
        img: Optional[np.ndarray] = None
        method = "none"

        try:
            img = self._screencap_socket()
            method = "socket"
        except Exception as socket_err:
            logger.warning(
                f"Direct socket screencap failed ({socket_err}). Attempting subprocess fallback..."
            )
            try:
                img = self._screencap_subprocess()
                method = "subprocess"
            except Exception as sub_err:
                logger.error(f"Subprocess screencap fallback failed: {sub_err}")
                raise RuntimeError(
                    f"All screencap methods failed. Socket: {socket_err} | Subprocess: {sub_err}"
                ) from sub_err

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_screencap_latency_ms = elapsed_ms
        self._last_capture_method = method

        logger.debug(f"Screencap captured via {method} in {elapsed_ms:.1f} ms, shape={img.shape}")
        return img

    def shell(self, cmd: str) -> str:
        """
        Execute an arbitrary shell command on the target Android device.
        Uses direct socket first, falling back to subprocess.
        """
        try:
            raw_out = self.execute_socket_command(f"exec:{cmd}")
            return raw_out.decode("utf-8", errors="replace")
        except Exception as socket_err:
            logger.debug(f"Direct socket shell execution failed ({socket_err}), trying subprocess...")
            if not self.adb_path:
                raise RuntimeError(f"Socket shell failed and no adb_path available: {socket_err}") from socket_err

            sub_cmd = [self.adb_path, "-s", self.serial, "shell", cmd]
            res = subprocess.run(
                sub_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.socket_timeout,
            )
            return res.stdout

    def swipe(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        duration_ms: int = 300,
        clamp_safe_release: bool = True,
    ) -> bool:
        """
        Dispatch a swipe / drag gesture from (x1, y1) to (x2, y2) with specified duration.
        
        Args:
            x1: Starting X coordinate.
            y1: Starting Y coordinate.
            x2: Ending X coordinate.
            y2: Ending Y coordinate.
            duration_ms: Duration of swipe in milliseconds.
            clamp_safe_release: If True, clamps y2 to safe release limit (Y <= 1580).
        
        Returns:
            bool: True if command executed without error.
        """
        # Clamp start coordinates strictly to physical bounds
        cx1, cy1 = clamp_coordinates(x1, y1)

        # Clamp end coordinates strictly to physical bounds
        cx2, cy2 = clamp_coordinates(x2, y2)

        # Apply safe release clamping to destination Y if requested
        if clamp_safe_release:
            cy2 = clamp_release_y(cy2)

        cmd = f"input swipe {cx1} {cy1} {cx2} {cy2} {int(duration_ms)}"
        logger.debug(f"Dispatching swipe: {cmd}")

        try:
            self.shell(cmd)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch swipe ({cmd}): {e}")
            return False

    def tap(self, x: Union[int, float], y: Union[int, float]) -> bool:
        """
        Dispatch a single touch tap at (x, y).
        
        Args:
            x: Touch X coordinate.
            y: Touch Y coordinate.
        
        Returns:
            bool: True if command executed without error.
        """
        cx, cy = clamp_coordinates(x, y)
        cmd = f"input tap {cx} {cy}"
        logger.debug(f"Dispatching tap: {cmd}")

        try:
            self.shell(cmd)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch tap ({cmd}): {e}")
            return False

    def is_connected(self) -> bool:
        """Check if ADB server daemon is reachable and target device is connected."""
        try:
            sock = self._create_socket()
            try:
                self._switch_device_transport(sock)
                return True
            finally:
                sock.close()
        except Exception:
            # Try subprocess fallback
            if self.adb_path:
                try:
                    res = subprocess.run(
                        [self.adb_path, "devices"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.connect_timeout,
                    )
                    return self.serial in res.stdout
                except Exception:
                    return False
            return False
