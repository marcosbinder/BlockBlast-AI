"""
conftest.py - Shared fixtures, test generators, mock ADB socket server,
and image loaders for Block Blast Autonomous Engine E2E Test Suite.
"""

import os
import sys
import io
import time
import socket
import threading
import pickle
from typing import Dict, List, Tuple, Optional, Any

import pytest
import cv2
import numpy as np
import neat

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from adb_client import (
    FastADBSocketClient,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BOARD_LEFT_X,
    BOARD_RIGHT_X,
    BOARD_TOP_Y,
    BOARD_BOTTOM_Y,
    TRAY_CANCEL_ZONE_START_Y,
    MAX_SAFE_RELEASE_Y,
)
from cv_detector import BlockBlastDetector, CANONICAL_SHAPES
from game import BlockBlast, Piece, BLOCK_SHAPES, PIECE_NAMES
import train

CHAMPION_PATH = os.path.join(WORKSPACE_DIR, "checkpoints", "best_champion.pkl")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "config-feedforward")


# =====================================================================
# REPOSITORY SAMPLE IMAGE FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def workspace_root() -> str:
    return WORKSPACE_DIR


@pytest.fixture(scope="session")
def repo_png_files(workspace_root: str) -> List[str]:
    """Finds all PNG screenshots available in the workspace."""
    pngs = [f for f in os.listdir(workspace_root) if f.lower().endswith(".png")]
    return sorted(pngs)


@pytest.fixture(scope="session")
def sample_images(workspace_root: str, repo_png_files: List[str]) -> Dict[str, np.ndarray]:
    """Loads all repository sample PNGs as BGR numpy arrays."""
    images = {}
    for filename in repo_png_files:
        path = os.path.join(workspace_root, filename)
        img = cv2.imread(path)
        if img is not None:
            images[filename] = img
    return images


# =====================================================================
# SYNTHETIC MULTI-THEME IMAGE GENERATOR
# =====================================================================

THEME_PALETTES = {
    "wood": {
        "screen_bg": (35, 43, 84),      # BGR
        "board_bg": (47, 71, 119),
        "board_recess": (40, 60, 100),
        "tray_bg": (82, 121, 181),
        "block_color": (30, 140, 220),   # Warm wood / gold
        "block_border": (10, 80, 150),
    },
    "blue": {
        "screen_bg": (20, 20, 40),
        "board_bg": (84, 43, 35),
        "board_recess": (70, 35, 28),
        "tray_bg": (148, 81, 58),
        "block_color": (230, 180, 40),   # Cyan gem
        "block_border": (180, 120, 20),
    },
    "neon": {
        "screen_bg": (10, 10, 15),
        "board_bg": (25, 25, 30),
        "board_recess": (18, 18, 22),
        "tray_bg": (35, 35, 45),
        "block_color": (50, 255, 50),    # Bright neon green
        "block_border": (20, 200, 20),
    },
    "jungle": {
        "screen_bg": (25, 50, 25),
        "board_bg": (35, 75, 40),
        "board_recess": (28, 60, 32),
        "tray_bg": (45, 90, 50),
        "block_color": (30, 200, 240),   # Amber gold
        "block_border": (20, 140, 180),
    },
}


class SyntheticFrameBuilder:
    """Creates realistic 1080x2400 game frames for deterministic testing."""

    def __init__(self, theme: str = "wood"):
        self.theme_name = theme
        self.palette = THEME_PALETTES.get(theme, THEME_PALETTES["wood"])
        self.width = 1080
        self.height = 2400

        # Layout
        self.board_left = 61
        self.board_top = 581
        self.board_right = 1018
        self.board_bottom = 1537
        self.cell_w = (self.board_right - self.board_left) / 8.0
        self.cell_h = (self.board_bottom - self.board_top) / 8.0

        self.tray_left = 60
        self.tray_right = 1020
        self.tray_top = 1660
        self.tray_bottom = 2050
        self.slot_centers_x = [220, 540, 860]

    def build_empty_frame(self) -> np.ndarray:
        """Builds a full 1080x2400 frame with empty board and empty tray."""
        frame = np.full((self.height, self.width, 3), self.palette["screen_bg"], dtype=np.uint8)

        # Draw outer board frame / bezel
        cv2.rectangle(
            frame,
            (self.board_left - 10, self.board_top - 10),
            (self.board_right + 10, self.board_bottom + 10),
            self.palette["board_recess"],
            thickness=-1,
        )
        # Draw board playable surface
        cv2.rectangle(
            frame,
            (self.board_left, self.board_top),
            (self.board_right, self.board_bottom),
            self.palette["board_bg"],
            thickness=-1,
        )
        # Paint bezel corner notches explicitly
        frame[self.board_top + 10:self.board_top + 25, self.board_left + 10:self.board_left + 25] = self.palette["board_recess"]
        frame[self.board_top + 10:self.board_top + 25, self.board_right - 25:self.board_right - 10] = self.palette["board_recess"]
        frame[self.board_bottom - 25:self.board_bottom - 10, self.board_left + 10:self.board_left + 25] = self.palette["board_recess"]
        frame[self.board_bottom - 25:self.board_bottom - 10, self.board_right - 25:self.board_right - 10] = self.palette["board_recess"]

        # Draw cell grid recesses
        for r in range(8):
            for c in range(8):
                cx = int(self.board_left + (c + 0.5) * self.cell_w)
                cy = int(self.board_top + (r + 0.5) * self.cell_h)
                x1, y1 = int(cx - self.cell_w * 0.44), int(cy - self.cell_h * 0.44)
                x2, y2 = int(cx + self.cell_w * 0.44), int(cy + self.cell_h * 0.44)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.palette["board_recess"], thickness=2)

        # Draw tray container
        cv2.rectangle(
            frame,
            (self.tray_left, self.tray_top),
            (self.tray_right, self.tray_bottom),
            self.palette["tray_bg"],
            thickness=-1,
        )
        return frame

    def draw_board_cells(self, frame: np.ndarray, occupied_cells: List[Tuple[int, int]], block_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Draws solid occupied blocks onto the 8x8 grid."""
        color = block_color or self.palette["block_color"]
        border = self.palette["block_border"]

        for r, c in occupied_cells:
            if not (0 <= r < 8 and 0 <= c < 8):
                continue
            cx = int(self.board_left + (c + 0.5) * self.cell_w)
            cy = int(self.board_top + (r + 0.5) * self.cell_h)
            x1, y1 = int(cx - self.cell_w * 0.40), int(cy - self.cell_h * 0.40)
            x2, y2 = int(cx + self.cell_w * 0.40), int(cy + self.cell_h * 0.40)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness=4)
            # Add bevel texture variance
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), thickness=-1)

        # Ensure bezel corner notches are preserved
        frame[self.board_top + 10:self.board_top + 25, self.board_left + 10:self.board_left + 25] = self.palette["board_recess"]
        frame[self.board_top + 10:self.board_top + 25, self.board_right - 25:self.board_right - 10] = self.palette["board_recess"]
        frame[self.board_bottom - 25:self.board_bottom - 10, self.board_left + 10:self.board_left + 25] = self.palette["board_recess"]
        frame[self.board_bottom - 25:self.board_bottom - 10, self.board_right - 25:self.board_right - 10] = self.palette["board_recess"]
        return frame

    def draw_tray_piece(self, frame: np.ndarray, slot_idx: int, piece_name: str) -> np.ndarray:
        """Draws a piece into the specified tray slot (0, 1, or 2)."""
        if piece_name not in BLOCK_SHAPES or not (0 <= slot_idx < 3):
            return frame

        blocks = BLOCK_SHAPES[piece_name]
        h = max(r for r, c in blocks) + 1
        w = max(c for r, c in blocks) + 1

        slot_cx = self.slot_centers_x[slot_idx]
        slot_cy = (self.tray_top + self.tray_bottom) // 2
        mini_size = 58.5

        piece_w_px = w * mini_size
        piece_h_px = h * mini_size

        origin_x = slot_cx - piece_w_px / 2.0
        origin_y = slot_cy - piece_h_px / 2.0

        for r, c in blocks:
            x1 = int(origin_x + c * mini_size + 1)
            y1 = int(origin_y + r * mini_size + 1)
            x2 = int(origin_x + (c + 1) * mini_size - 1)
            y2 = int(origin_y + (r + 1) * mini_size - 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.palette["block_color"], thickness=-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.palette["block_border"], thickness=2)

        return frame

    def draw_ghost_highlights(self, frame: np.ndarray, ghost_cells: List[Tuple[int, int]], delta_e: float = 40.0) -> np.ndarray:
        """Applies a translucent ghost highlight to target cells with exact Euclidean color distance."""
        bg_col = np.array(self.palette["board_bg"], dtype=np.float32)
        # Shift each BGR channel by delta_e / sqrt(3) to achieve exact Euclidean distance ||delta|| = delta_e
        channel_shift = delta_e / np.sqrt(3.0)
        ghost_col = np.clip(bg_col + channel_shift, 0, 255).astype(np.uint8)

        for r, c in ghost_cells:
            if not (0 <= r < 8 and 0 <= c < 8):
                continue
            cx = int(self.board_left + (c + 0.5) * self.cell_w)
            cy = int(self.board_top + (r + 0.5) * self.cell_h)
            x1, y1 = int(cx - self.cell_w * 0.42), int(cy - self.cell_h * 0.42)
            x2, y2 = int(cx + self.cell_w * 0.42), int(cy + self.cell_h * 0.42)
            cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(int(v) for v in ghost_col), thickness=-1)
        return frame


@pytest.fixture
def frame_builder() -> SyntheticFrameBuilder:
    return SyntheticFrameBuilder(theme="wood")


# =====================================================================
# MOCK ADB SOCKET SERVER
# =====================================================================

class MockADBServer:
    """Threaded TCP server mimicking ADB server daemon protocol at 127.0.0.1."""

    def __init__(self, serial: str = "ZF524K4RCM"):
        self.serial = serial
        self.sock: Optional[socket.socket] = None
        self.port: int = 0
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.received_commands: List[str] = []
        self.mock_png_bytes: bytes = b""
        self.lock = threading.Lock()

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(10)
        self.port = self.sock.getsockname()[1]
        self.running = True

        # Generate a lightweight dummy 1080x2400 PNG
        dummy_img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", dummy_img)
        self.mock_png_bytes = encoded.tobytes()

        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self):
        while self.running and self.sock:
            try:
                conn, _ = self.sock.accept()
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except Exception:
                break

    def _handle_client(self, conn: socket.socket):
        conn.settimeout(3.0)
        try:
            # 1. Read transport request prefix
            len_hex = conn.recv(4)
            if not len_hex:
                conn.close()
                return
            cmd_len = int(len_hex.decode("ascii"), 16)
            cmd = conn.recv(cmd_len).decode("utf-8")

            with self.lock:
                self.received_commands.append(cmd)

            if cmd == f"host:transport:{self.serial}":
                conn.sendall(b"OKAY")
            else:
                conn.sendall(b"FAIL0012device not found")
                conn.close()
                return

            # 2. Read service request
            len_hex = conn.recv(4)
            if not len_hex:
                conn.close()
                return
            cmd_len = int(len_hex.decode("ascii"), 16)
            service_cmd = conn.recv(cmd_len).decode("utf-8")

            with self.lock:
                self.received_commands.append(service_cmd)

            conn.sendall(b"OKAY")

            # 3. Handle service command response
            if service_cmd == "exec:screencap -p":
                conn.sendall(self.mock_png_bytes)
            elif service_cmd.startswith("exec:input"):
                # Echo / empty output
                pass
            elif service_cmd.startswith("exec:"):
                conn.sendall(b"output from mock exec")

        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=1.0)


@pytest.fixture
def mock_adb_server() -> MockADBServer:
    server = MockADBServer(serial="ZF524K4RCM")
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_adb_client(mock_adb_server: MockADBServer) -> FastADBSocketClient:
    return FastADBSocketClient(
        serial=mock_adb_server.serial,
        host="127.0.0.1",
        port=mock_adb_server.port,
        connect_timeout=1.0,
        socket_timeout=2.0,
    )


# =====================================================================
# VISION DETECTOR, NEURAL NET, AND SIMULATOR FIXTURES
# =====================================================================

@pytest.fixture
def detector() -> BlockBlastDetector:
    return BlockBlastDetector()


@pytest.fixture(scope="session")
def neat_champion_net() -> Optional[neat.nn.FeedForwardNetwork]:
    """Loads champion neural network if checkpoint exists."""
    if not os.path.exists(CHAMPION_PATH):
        return None
    try:
        with open(CHAMPION_PATH, "rb") as f:
            data = pickle.load(f)
        config = train._get_worker_config(CONFIG_PATH)
        net = neat.nn.FeedForwardNetwork.create(data["genome"], config)
        return net
    except Exception as e:
        print(f"Warning: could not load NEAT champion network: {e}")
        return None


@pytest.fixture
def game_sim() -> BlockBlast:
    """Deterministic game simulation initialized with seed 42."""
    return BlockBlast(seed=42)
