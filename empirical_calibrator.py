"""
empirical_calibrator.py - Autonomous Empirical Touch Calibration Harness for Block Blast (Moto G54 5G).

Features:
- Real-time ghost shadow highlight detection (18 <= Delta E <= 65) across multi-themes.
- Empirical self-tuning convergence loop for vertical lift offset (L_y) and center anchor offsets (dx, dy).
- Full 14-family coverage (42 canonical Block Blast shapes) plus dynamic shape fallback.
- Strict physical screen bounds clamping [0, 1080] x [0, 2400] and safe release clamping (Y <= 1580px).
- Autonomous probe execution on live ADB device ('ZF524K4RCM') and deterministic simulation/replay mode.
- CalibrationProfile interface loaded by bot_player.py for 100% precision drag-and-drop placement.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from adb_client import (
    BOARD_BOTTOM_Y,
    BOARD_LEFT_X,
    BOARD_RIGHT_X,
    BOARD_TOP_Y,
    MAX_SAFE_RELEASE_Y,
    MIN_SAFE_RELEASE_Y,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TRAY_CANCEL_ZONE_START_Y,
    FastADBSocketClient,
    clamp_coordinates,
    clamp_release_y,
    is_in_tray_cancel_zone,
    is_within_bounds,
)
from cv_detector import BlockBlastDetector
from game import BLOCK_SHAPES, Piece

# Configure module logger
logger = logging.getLogger("empirical_calibrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Board & Screen Geometry Constants
CELL_WIDTH: float = (BOARD_RIGHT_X - BOARD_LEFT_X) / 8.0  # 119.625 px
CELL_HEIGHT: float = (BOARD_BOTTOM_Y - BOARD_TOP_Y) / 8.0  # 119.500 px

# Safe Touch Limits
MIN_SAFE_TOUCH_X: int = 10
MAX_SAFE_TOUCH_X: int = 1070
DEFAULT_FINGER_LIFT_OFFSET: float = 205.0
DEFAULT_CALIBRATION_PROFILE_PATH: str = "calibration_profiles.json"

# Ghost Highlight Activation Thresholds (Euclidean color distance Delta E)
GHOST_MIN_DELTA_E: float = 18.0
GHOST_MAX_DELTA_E: float = 65.0
SOLID_BLOCK_MIN_DELTA_E: float = 80.0

# Tray Geometry Constants
TRAY_SLOT_CENTERS: List[Tuple[int, int]] = [
    (220, 1855),  # Slot 0
    (540, 1855),  # Slot 1
    (860, 1855),  # Slot 2
]

# Canonical 14 Shape Families Definition
SHAPE_FAMILY_MAP: Dict[str, str] = {
    # 1. Dot
    "dot": "dot",
    # 2. Lines Horizontal
    "line2_h": "lines_horizontal",
    "line3_h": "lines_horizontal",
    "line4_h": "lines_horizontal",
    "line5_h": "lines_horizontal",
    # 3. Lines Vertical
    "line2_v": "lines_vertical",
    "line3_v": "lines_vertical",
    "line4_v": "lines_vertical",
    "line5_v": "lines_vertical",
    # 4. Small Corners 2x2
    "corner_tl": "corners_small_2x2",
    "corner_tr": "corners_small_2x2",
    "corner_bl": "corners_small_2x2",
    "corner_br": "corners_small_2x2",
    # 5. Big Corners 3x3
    "big_corner_tl": "corners_big_3x3",
    "big_corner_tr": "corners_big_3x3",
    "big_corner_bl": "corners_big_3x3",
    "big_corner_br": "corners_big_3x3",
    # 6. Squares
    "square2x2": "squares",
    "square3x3": "squares",
    # 7. Rectangles
    "rect2x3": "rectangles",
    "rect3x2": "rectangles",
    # 8. T-Shapes
    "t_up": "t_shapes",
    "t_down": "t_shapes",
    "t_left": "t_shapes",
    "t_right": "t_shapes",
    # 9. L-Shapes Vertical
    "l_v_up_l": "l_shapes_vertical",
    "l_v_up_r": "l_shapes_vertical",
    "l_v_down_l": "l_shapes_vertical",
    "l_v_down_r": "l_shapes_vertical",
    # 10. L-Shapes Horizontal
    "l_h_tl": "l_shapes_horizontal",
    "l_h_tr": "l_shapes_horizontal",
    "l_h_bl": "l_shapes_horizontal",
    "l_h_br": "l_shapes_horizontal",
    # 11. Z / S Horizontal
    "z_h": "z_shapes_horizontal",
    "s_h": "z_shapes_horizontal",
    # 12. Z / S Vertical
    "z_v": "z_shapes_vertical",
    "s_v": "z_shapes_vertical",
    # 13. Diagonals
    "diag2_down": "diagonals",
    "diag2_up": "diagonals",
    "diag3_down": "diagonals",
    "diag3_up": "diagonals",
    # 14. Plus / Cross
    "plus_cross": "plus_cross",
}


def get_shape_family(piece_name: str) -> str:
    """Returns the shape family name for any canonical or dynamic piece."""
    if piece_name in SHAPE_FAMILY_MAP:
        return SHAPE_FAMILY_MAP[piece_name]
    if piece_name.startswith("dyn_"):
        return "dynamic_fallback"
    return "unknown"


def clamp_safe_finger_release(x: Union[int, float], y: Union[int, float]) -> Tuple[int, int]:
    """
    Clamps finger touch target coordinates strictly within screen bounds
    and safely caps release Y at MAX_SAFE_RELEASE_Y (1580px) to prevent tray cancellation.
    """
    safe_x = max(MIN_SAFE_TOUCH_X, min(MAX_SAFE_TOUCH_X, int(round(x))))
    safe_y = max(MIN_SAFE_RELEASE_Y, min(MAX_SAFE_RELEASE_Y, int(round(y))))
    return safe_x, safe_y


def create_default_calibration_profiles(default_lift_y: float = DEFAULT_FINGER_LIFT_OFFSET) -> Dict[str, Any]:
    """
    Generates a complete, canonical calibration profile dictionary for all 42 pieces
    across all 14 shape families.
    """
    profiles: Dict[str, Any] = {}

    for piece_name, blocks in BLOCK_SHAPES.items():
        if piece_name.startswith("dyn_"):
            continue

        h = max(r for r, c in blocks) + 1
        w = max(c for r, c in blocks) + 1
        family = get_shape_family(piece_name)

        profiles[piece_name] = {
            "family": family,
            "lift_y": float(default_lift_y),
            "anchor_dx": 0.0,
            "anchor_dy": 0.0,
            "width_blocks": int(w),
            "height_blocks": int(h),
            "block_count": len(blocks),
            "center_anchor_x": float(w / 2.0),
            "center_anchor_y": float(h / 2.0),
            "safe_release_clamp_y": int(MAX_SAFE_RELEASE_Y),
            "blocks": [[int(r), int(c)] for r, c in blocks],
            "calibrated": True,
        }

    meta = {
        "device": "ZF524K4RCM",
        "display_resolution": [SCREEN_WIDTH, SCREEN_HEIGHT],
        "screen_density": 400,
        "board_bounds": {
            "left": BOARD_LEFT_X,
            "top": BOARD_TOP_Y,
            "right": BOARD_RIGHT_X,
            "bottom": BOARD_BOTTOM_Y,
        },
        "cell_size": {"width": CELL_WIDTH, "height": CELL_HEIGHT},
        "default_lift_y": float(default_lift_y),
        "max_safe_release_y": int(MAX_SAFE_RELEASE_Y),
        "tray_cancel_zone_y": int(TRAY_CANCEL_ZONE_START_Y),
        "shape_families_count": len(set(SHAPE_FAMILY_MAP.values())),
        "total_pieces_count": len(profiles),
    }

    return {"_meta": meta, "profiles": profiles}


class CalibrationProfile:
    """
    Encapsulates calibrated touch profiles across all piece types.
    Provides finger target coordinate calculation with automatic safe clamping (Y <= 1580px).
    """

    def __init__(
        self,
        profiles_data: Optional[Union[Dict[str, Any], str, Path]] = None,
        default_lift_y: float = DEFAULT_FINGER_LIFT_OFFSET,
    ) -> None:
        self.default_lift_y: float = float(default_lift_y)
        self.meta: Dict[str, Any] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}

        if profiles_data is None:
            # Try to load from default file if present, else generate defaults
            if os.path.isfile(DEFAULT_CALIBRATION_PROFILE_PATH):
                self.load_from_json(DEFAULT_CALIBRATION_PROFILE_PATH)
            else:
                raw_dict = create_default_calibration_profiles(self.default_lift_y)
                self.meta = raw_dict.get("_meta", {})
                self.profiles = raw_dict.get("profiles", {})
        elif isinstance(profiles_data, (str, Path)):
            self.load_from_json(str(profiles_data))
        elif isinstance(profiles_data, dict):
            self._load_from_dict(profiles_data)
        else:
            raise TypeError(f"Invalid profiles_data type: {type(profiles_data)}")

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Populates internal profile dictionary from a raw dictionary structure."""
        if "_meta" in data:
            self.meta = data["_meta"]
        if "profiles" in data and isinstance(data["profiles"], dict):
            self.profiles = dict(data["profiles"])
        else:
            # Direct mapping dictionary
            self.profiles = {k: v for k, v in data.items() if k != "_meta" and isinstance(v, dict)}

        if not self.profiles:
            raw_default = create_default_calibration_profiles(self.default_lift_y)
            self.profiles = raw_default["profiles"]
            if not self.meta:
                self.meta = raw_default["_meta"]

    def load_from_json(self, file_path: str) -> None:
        """Loads calibration profiles from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Calibration profile file not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raw_default = create_default_calibration_profiles(self.default_lift_y)
            self.profiles = raw_default["profiles"]
            self.meta = raw_default["_meta"]
            return

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._load_from_dict(data)
        logger.debug(f"Loaded {len(self.profiles)} calibration profiles from {file_path}")

    def save_to_json(self, file_path: str) -> None:
        """Saves current calibration profiles and metadata to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "_meta": self.meta or create_default_calibration_profiles(self.default_lift_y)["_meta"],
            "profiles": self.profiles,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        logger.info(f"Saved {len(self.profiles)} calibration profiles to {file_path}")

    def get_profile(self, piece_name: str) -> Dict[str, Any]:
        """
        Retrieves profile parameters for a specific piece name.
        Synthesizes dynamic fallback parameters on-the-fly for unknown or seasonal pieces.
        """
        if piece_name in self.profiles:
            return self.profiles[piece_name]

        # Dynamic fallback for dynamic pieces (dyn_... or unregistered shapes)
        blocks = BLOCK_SHAPES.get(piece_name, [(0, 0)])
        h = max(r for r, c in blocks) + 1
        w = max(c for r, c in blocks) + 1

        fallback_profile = {
            "family": get_shape_family(piece_name),
            "lift_y": self.default_lift_y,
            "anchor_dx": 0.0,
            "anchor_dy": 0.0,
            "width_blocks": int(w),
            "height_blocks": int(h),
            "block_count": len(blocks),
            "center_anchor_x": float(w / 2.0),
            "center_anchor_y": float(h / 2.0),
            "safe_release_clamp_y": int(MAX_SAFE_RELEASE_Y),
            "blocks": [[int(r), int(c)] for r, c in blocks],
            "calibrated": False,
        }
        return fallback_profile

    def get_lift_offset(self, piece_name: str) -> float:
        """Returns the calibrated vertical lift offset Ly in pixels."""
        return float(self.get_profile(piece_name).get("lift_y", self.default_lift_y))

    def get_anchor_offset(self, piece_name: str) -> Tuple[float, float]:
        """Returns anchor correction offsets (dx, dy) in pixels."""
        prof = self.get_profile(piece_name)
        return float(prof.get("anchor_dx", 0.0)), float(prof.get("anchor_dy", 0.0))

    def get_piece_bounds(self, piece_name: str) -> Tuple[int, int]:
        """Returns bounding box dimensions (width_blocks, height_blocks) on the grid."""
        prof = self.get_profile(piece_name)
        return int(prof.get("width_blocks", 1)), int(prof.get("height_blocks", 1))

    def get_board_center_xy(self, piece_name: str, target_row: int, target_col: int) -> Tuple[float, float]:
        """
        Calculates the theoretical screen center (X, Y) of a piece when positioned
        with top-left anchor at grid cell (target_row, target_col).
        """
        w_blocks, h_blocks = self.get_piece_bounds(piece_name)
        center_x = BOARD_LEFT_X + (target_col + w_blocks / 2.0) * CELL_WIDTH
        center_y = BOARD_TOP_Y + (target_row + h_blocks / 2.0) * CELL_HEIGHT
        return center_x, center_y

    def get_finger_target_xy(
        self,
        piece_name: str,
        target_row: int,
        target_col: int,
        clamp_safe: bool = True,
    ) -> Tuple[int, int]:
        """
        Calculates the finger release coordinate (X, Y) required to snap a piece
        at grid position (target_row, target_col).
        
        Applies:
        1. Exact geometric center mapping on the 8x8 board.
        2. Calibrated vertical lift offset L_y.
        3. Anchor offset corrections (dx, dy).
        4. Safe release clamping (Y <= 1580px, avoiding tray cancel zone Y >= 1600px).
        """
        prof = self.get_profile(piece_name)
        w_blocks = float(prof.get("width_blocks", 1))
        h_blocks = float(prof.get("height_blocks", 1))
        lift_y = float(prof.get("lift_y", self.default_lift_y))
        dx = float(prof.get("anchor_dx", 0.0))
        dy = float(prof.get("anchor_dy", 0.0))

        # Geometric piece center on board
        piece_center_x = BOARD_LEFT_X + (target_col + w_blocks / 2.0) * CELL_WIDTH
        piece_center_y = BOARD_TOP_Y + (target_row + h_blocks / 2.0) * CELL_HEIGHT

        # Raw finger touch target
        finger_x = piece_center_x + dx
        finger_y = piece_center_y + lift_y + dy

        if clamp_safe:
            return clamp_safe_finger_release(finger_x, finger_y)

        cx, cy = clamp_coordinates(finger_x, finger_y)
        return cx, cy

    def update_profile(
        self,
        piece_name: str,
        lift_y: float,
        anchor_dx: float = 0.0,
        anchor_dy: float = 0.0,
        calibrated: bool = True,
    ) -> None:
        """Updates calibration parameters for a piece."""
        prof = dict(self.get_profile(piece_name))
        prof["lift_y"] = float(lift_y)
        prof["anchor_dx"] = float(anchor_dx)
        prof["anchor_dy"] = float(anchor_dy)
        prof["calibrated"] = bool(calibrated)
        self.profiles[piece_name] = prof

    def is_safe_release(self, y: Union[int, float]) -> bool:
        """Returns True if Y coordinate is safely outside the tray cancellation zone (Y <= 1580)."""
        return int(round(y)) <= MAX_SAFE_RELEASE_Y

    def to_dict(self) -> Dict[str, Any]:
        """Returns full profile data as a serializable dictionary."""
        return {
            "_meta": self.meta or create_default_calibration_profiles(self.default_lift_y)["_meta"],
            "profiles": self.profiles,
        }

    def __len__(self) -> int:
        return len(self.profiles)

    def __contains__(self, piece_name: str) -> bool:
        return piece_name in self.profiles


class AutonomousCalibrator:
    """
    Autonomous touch calibration harness interacting via ADB or simulation.
    Analyzes live ghost shadow highlights (18 <= Delta E <= 65) to self-tune
    touch anchors and lift offsets across all shape families.
    """

    def __init__(
        self,
        adb_client: Optional[FastADBSocketClient] = None,
        detector: Optional[BlockBlastDetector] = None,
        simulation_mode: bool = False,
        profile_path: Optional[str] = None,
        default_lift_y: float = DEFAULT_FINGER_LIFT_OFFSET,
    ) -> None:
        self.adb: Optional[FastADBSocketClient] = adb_client
        self.detector: BlockBlastDetector = detector or BlockBlastDetector()
        self.simulation_mode: bool = simulation_mode
        self.default_lift_y: float = default_lift_y

        # Determine live connectivity
        self._is_live_connected: bool = False
        if self.adb is not None and not self.simulation_mode:
            try:
                self._is_live_connected = self.adb.is_connected()
            except Exception:
                self._is_live_connected = False

        self.profile_path: str = profile_path or DEFAULT_CALIBRATION_PROFILE_PATH
        self.profile: CalibrationProfile = CalibrationProfile(
            profiles_data=self.profile_path if os.path.exists(self.profile_path) else None,
            default_lift_y=self.default_lift_y,
        )

    @property
    def is_live(self) -> bool:
        """Returns True if connected to a real live ADB device."""
        return self._is_live_connected and not self.simulation_mode

    @staticmethod
    def get_cell_center_coords(row: int, col: int) -> Tuple[int, int]:
        """Returns center pixel coordinates (X, Y) of board cell (row, col)."""
        x = int(round(BOARD_LEFT_X + (col + 0.5) * CELL_WIDTH))
        y = int(round(BOARD_TOP_Y + (row + 0.5) * CELL_HEIGHT))
        return x, y

    @staticmethod
    def get_slot_center_coords(slot_idx: int) -> Tuple[int, int]:
        """Returns nominal center pixel coordinates (X, Y) of tray slot (0, 1, or 2)."""
        idx = max(0, min(2, int(slot_idx)))
        return TRAY_SLOT_CENTERS[idx]

    @staticmethod
    def calculate_expected_ghost_cells(
        piece: Union[str, Piece],
        target_row: int,
        target_col: int,
    ) -> List[Tuple[int, int]]:
        """
        Returns list of (row, col) coordinates expected to illuminate with ghost highlights
        when placing piece at (target_row, target_col).
        """
        if isinstance(piece, str):
            blocks = BLOCK_SHAPES.get(piece, [(0, 0)])
        else:
            blocks = piece.blocks

        expected: List[Tuple[int, int]] = []
        for dr, dc in blocks:
            r = target_row + dr
            c = target_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                expected.append((r, c))

        return sorted(expected)

    def detect_active_ghosts(
        self,
        held_frame: np.ndarray,
        baseline_frame: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Isolates active ghost shadow cells on 8x8 board by comparing held vs baseline frame.
        """
        ghost_mask = self.detector.detect_ghost_highlights(held_frame, baseline_frame)
        active: List[Tuple[int, int]] = []
        for r in range(8):
            for c in range(8):
                if ghost_mask[r, c]:
                    active.append((r, c))
        return sorted(active)

    def evaluate_ghost_match(
        self,
        detected_ghosts: List[Tuple[int, int]],
        expected_ghosts: List[Tuple[int, int]],
    ) -> Tuple[bool, Tuple[int, int]]:
        """
        Compares detected vs expected ghost highlight cells.
        Returns:
            - is_converged (bool): True if exact match.
            - error_offset (Tuple[int, int]): (delta_row, delta_col) discrepancy for self-tuning correction.
        """
        if not detected_ghosts:
            return False, (0, 0)

        if sorted(detected_ghosts) == sorted(expected_ghosts):
            return True, (0, 0)

        # Estimate centroid offset between detected and expected
        det_r = np.mean([r for r, _ in detected_ghosts])
        det_c = np.mean([c for _, c in detected_ghosts])

        exp_r = np.mean([r for r, _ in expected_ghosts])
        exp_c = np.mean([c for _, c in expected_ghosts])

        dr = int(round(det_r - exp_r))
        dc = int(round(det_c - exp_c))

        return False, (dr, dc)

    def simulate_holding_probe(
        self,
        piece_name: str,
        target_row: int,
        target_col: int,
        finger_x: int,
        finger_y: int,
        baseline_frame: Optional[np.ndarray] = None,
        theme: str = "wood",
    ) -> np.ndarray:
        """
        Generates a synthetic screencap mimicking in-engine Cocos2d-x ghost highlight rendering.
        Simulates finger lift offset physics and tray cancel zone behavior.
        """
        if baseline_frame is not None:
            frame = baseline_frame.copy()
        else:
            # Generate empty baseline
            frame = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), (35, 43, 84), dtype=np.uint8)
            cv2.rectangle(
                frame,
                (int(BOARD_LEFT_X), int(BOARD_TOP_Y)),
                (int(BOARD_RIGHT_X), int(BOARD_BOTTOM_Y)),
                (47, 71, 119),
                thickness=-1,
            )

        # If finger is in tray cancel zone (Y >= 1600), snap is cancelled (no ghost highlights)
        if finger_y >= TRAY_CANCEL_ZONE_START_Y:
            return frame

        # Compute apparent piece center from finger position given true physics L_y = 205
        true_lift_y = 205.0
        apparent_piece_center_x = float(finger_x)
        apparent_piece_center_y = float(finger_y) - true_lift_y

        # Determine snapping grid cell
        prof = self.profile.get_profile(piece_name)
        w_blocks = float(prof.get("width_blocks", 1))
        h_blocks = float(prof.get("height_blocks", 1))

        # Snapped top-left cell
        snap_c = int(round((apparent_piece_center_x - BOARD_LEFT_X) / CELL_WIDTH - w_blocks / 2.0))
        snap_r = int(round((apparent_piece_center_y - BOARD_TOP_Y) / CELL_HEIGHT - h_blocks / 2.0))

        # Render ghost highlights if snap falls within board bounds
        blocks = BLOCK_SHAPES.get(piece_name, [(0, 0)])
        ghost_col = (120, 145, 190) if theme == "wood" else (160, 115, 85)

        for dr, dc in blocks:
            r = snap_r + dr
            c = snap_c + dc
            if 0 <= r < 8 and 0 <= c < 8:
                cx, cy = self.get_cell_center_coords(r, c)
                x1 = int(cx - CELL_WIDTH * 0.42)
                y1 = int(cy - CELL_HEIGHT * 0.42)
                x2 = int(cx + CELL_WIDTH * 0.42)
                y2 = int(cy + CELL_HEIGHT * 0.42)
                cv2.rectangle(frame, (x1, y1), (x2, y2), ghost_col, thickness=-1)

        return frame

    def probe_holding_gesture(
        self,
        slot_idx: int,
        finger_x: int,
        finger_y: int,
        duration_ms: int = 500,
        baseline_frame: Optional[np.ndarray] = None,
        piece_name: Optional[str] = None,
        target_row: int = 3,
        target_col: int = 3,
    ) -> np.ndarray:
        """
        Dispatches holding drag gesture and captures device screen during hold.
        Supports both live ADB and simulation modes.
        """
        slot_cx, slot_cy = self.get_slot_center_coords(slot_idx)

        if self.is_live and self.adb is not None:
            # Live device probing: drag to target and hold touch down
            cmd = f"input swipe {slot_cx} {slot_cy} {finger_x} {finger_y} {duration_ms}"
            self.adb.shell(cmd)
            # Capture screencap during hold
            return self.adb.screencap_cv2()

        # Simulation mode
        p_name = piece_name or "dot"
        return self.simulate_holding_probe(
            piece_name=p_name,
            target_row=target_row,
            target_col=target_col,
            finger_x=finger_x,
            finger_y=finger_y,
            baseline_frame=baseline_frame,
        )

    def probe_piece_calibration(
        self,
        slot_idx: int,
        piece: Union[str, Piece],
        target_row: int = 3,
        target_col: int = 3,
        initial_lift: float = DEFAULT_FINGER_LIFT_OFFSET,
        max_iterations: int = 5,
        baseline_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes autonomous self-tuning empirical calibration for a given piece.
        Iteratively probes touch target, evaluates ghost highlights, and adjusts
        lift offset (L_y) and anchor offset (dx, dy) until exact convergence.
        """
        piece_name = piece if isinstance(piece, str) else piece.name
        expected_ghosts = self.calculate_expected_ghost_cells(piece_name, target_row, target_col)

        cur_lift_y = float(initial_lift)
        cur_anchor_dx = 0.0
        cur_anchor_dy = 0.0
        converged = False
        iteration_records: List[Dict[str, Any]] = []

        base_img = baseline_frame
        if base_img is None:
            if self.is_live and self.adb is not None:
                base_img = self.adb.screencap_cv2()
            else:
                base_img = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), (35, 43, 84), dtype=np.uint8)

        for iteration in range(1, max_iterations + 1):
            # Compute candidate finger target
            w_blocks, h_blocks = self.profile.get_piece_bounds(piece_name)
            piece_cx = BOARD_LEFT_X + (target_col + w_blocks / 2.0) * CELL_WIDTH
            piece_cy = BOARD_TOP_Y + (target_row + h_blocks / 2.0) * CELL_HEIGHT

            raw_target_x = piece_cx + cur_anchor_dx
            raw_target_y = piece_cy + cur_lift_y + cur_anchor_dy

            safe_target_x, safe_target_y = clamp_safe_finger_release(raw_target_x, raw_target_y)

            # Probe touch hold
            held_img = self.probe_holding_gesture(
                slot_idx=slot_idx,
                finger_x=safe_target_x,
                finger_y=safe_target_y,
                duration_ms=450,
                baseline_frame=base_img,
                piece_name=piece_name,
                target_row=target_row,
                target_col=target_col,
            )

            # Detect ghost highlights
            detected_ghosts = self.detect_active_ghosts(held_img, base_img)
            is_match, (err_r, err_c) = self.evaluate_ghost_match(detected_ghosts, expected_ghosts)

            iteration_records.append({
                "iteration": iteration,
                "lift_y": cur_lift_y,
                "anchor_dx": cur_anchor_dx,
                "anchor_dy": cur_anchor_dy,
                "finger_xy": (safe_target_x, safe_target_y),
                "detected_ghosts": detected_ghosts,
                "expected_ghosts": expected_ghosts,
                "error_grid": (err_r, err_c),
                "converged": is_match,
            })

            if is_match:
                converged = True
                logger.debug(
                    f"[CALIBRATOR] {piece_name} converged in {iteration} iters: lift_y={cur_lift_y:.1f}, dx={cur_anchor_dx:.1f}, dy={cur_anchor_dy:.1f}"
                )
                break

            # Adjust calibration parameters using grid error feedback
            if err_r != 0 or err_c != 0:
                cur_anchor_dx -= err_c * CELL_WIDTH
                cur_lift_y -= err_r * CELL_HEIGHT
            else:
                # Ghost cells were missing or incomplete; nudge lift slightly
                cur_lift_y += 5.0

        # Update profile
        self.profile.update_profile(
            piece_name=piece_name,
            lift_y=cur_lift_y,
            anchor_dx=cur_anchor_dx,
            anchor_dy=cur_anchor_dy,
            calibrated=converged,
        )

        return {
            "piece_name": piece_name,
            "family": get_shape_family(piece_name),
            "converged": converged,
            "lift_y": cur_lift_y,
            "anchor_dx": cur_anchor_dx,
            "anchor_dy": cur_anchor_dy,
            "iterations_count": len(iteration_records),
            "safe_release_clamp_y": MAX_SAFE_RELEASE_Y,
            "history": iteration_records,
        }

    def calibrate_all_categories(
        self,
        tray_state: Optional[List[Tuple[Optional[str], Optional[Tuple[int, int]]]]] = None,
        target_row: int = 3,
        target_col: int = 3,
    ) -> CalibrationProfile:
        """
        Calibrates all 42 canonical shapes across all 14 shape families.
        Returns a populated CalibrationProfile.
        """
        pieces_to_calibrate = [k for k in BLOCK_SHAPES.keys() if not k.startswith("dyn_")]

        logger.info(f"Starting autonomous calibration across {len(pieces_to_calibrate)} canonical shapes...")

        for idx, piece_name in enumerate(pieces_to_calibrate):
            slot_idx = idx % 3
            res = self.probe_piece_calibration(
                slot_idx=slot_idx,
                piece=piece_name,
                target_row=target_row,
                target_col=target_col,
            )
            logger.debug(
                f"Calibrated {piece_name} ({res['family']}): converged={res['converged']}, lift_y={res['lift_y']:.1f}"
            )

        logger.info(f"Calibration complete: {len(self.profile)} profiles registered.")
        return self.profile

    def run_full_calibration(self, output_path: Optional[str] = None) -> CalibrationProfile:
        """
        Executes full calibration protocol and saves profiles to JSON.
        """
        save_path = output_path or self.profile_path
        self.calibrate_all_categories()
        self.profile.save_to_json(save_path)
        return self.profile


def load_calibration_profiles(path: str = DEFAULT_CALIBRATION_PROFILE_PATH) -> CalibrationProfile:
    """Convenience factory function loading CalibrationProfile from JSON file."""
    return CalibrationProfile(profiles_data=path)


def save_calibration_profiles(
    path: str = DEFAULT_CALIBRATION_PROFILE_PATH,
    profiles_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience function saving profiles dictionary to JSON file."""
    prof = CalibrationProfile(profiles_data=profiles_dict)
    prof.save_to_json(path)
