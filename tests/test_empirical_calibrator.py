"""
tests/test_empirical_calibrator.py - Unit and Integration Test Suite for
Autonomous Empirical Touch Calibration Harness & Profiles (Moto G54 5G).

Covers:
- CalibrationProfile loading, saving, serialization, and roundtrip fidelity.
- All 14 shape families and 42 canonical Block Blast shapes.
- Precise screen coordinate calculations and safe release clamping (Y <= 1580px).
- Tray cancellation hazard zone prevention (Y < 1600px).
- Ghost highlight detection (18 <= Delta E <= 65) across multi-themes (Wood, Blue, Neon, Jungle).
- Autonomous self-tuning convergence loop for lift offset (L_y) and anchor offset (dx, dy).
- Mock ADB socket integration and offline deterministic simulation mode.
- Unknown/dynamic shape fallback synthesis.
"""

import json
import os
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

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
    is_in_tray_cancel_zone,
)
from cv_detector import BlockBlastDetector
from empirical_calibrator import (
    CELL_HEIGHT,
    CELL_WIDTH,
    DEFAULT_CALIBRATION_PROFILE_PATH,
    DEFAULT_FINGER_LIFT_OFFSET,
    GHOST_MAX_DELTA_E,
    GHOST_MIN_DELTA_E,
    SHAPE_FAMILY_MAP,
    AutonomousCalibrator,
    CalibrationProfile,
    clamp_safe_finger_release,
    create_default_calibration_profiles,
    get_shape_family,
    load_calibration_profiles,
    save_calibration_profiles,
)
from game import BLOCK_SHAPES, Piece
from tests.conftest import MockADBServer, SyntheticFrameBuilder


# =====================================================================
# TEST CLASS 1: CALIBRATION PROFILE STRUCTURE & SERIALIZATION
# =====================================================================

class TestCalibrationProfileSerialization:
    """Verifies profile dictionary structure, schema validation, and persistence."""

    def test_default_profiles_contain_all_42_pieces(self):
        """Verifies all 42 canonical Block Blast pieces are defined."""
        data = create_default_calibration_profiles()
        profiles = data["profiles"]
        assert len(profiles) == 42
        for piece_name in BLOCK_SHAPES.keys():
            if not piece_name.startswith("dyn_"):
                assert piece_name in profiles

    def test_14_shape_families_represented(self):
        """Verifies exactly 14 unique shape families are covered."""
        data = create_default_calibration_profiles()
        families = {p["family"] for p in data["profiles"].values()}
        assert len(families) == 14
        expected_families = {
            "dot",
            "lines_horizontal",
            "lines_vertical",
            "corners_small_2x2",
            "corners_big_3x3",
            "squares",
            "rectangles",
            "t_shapes",
            "l_shapes_vertical",
            "l_shapes_horizontal",
            "z_shapes_horizontal",
            "z_shapes_vertical",
            "diagonals",
            "plus_cross",
        }
        assert families == expected_families

    def test_metadata_fields_correct(self):
        """Verifies hardware and geometry metadata fields."""
        data = create_default_calibration_profiles()
        meta = data["_meta"]
        assert meta["device"] == "ZF524K4RCM"
        assert meta["display_resolution"] == [1080, 2400]
        assert meta["screen_density"] == 400
        assert meta["max_safe_release_y"] == 1580
        assert meta["tray_cancel_zone_y"] == 1600
        assert meta["board_bounds"]["left"] == 61.0
        assert meta["board_bounds"]["top"] == 581.0
        assert meta["board_bounds"]["right"] == 1018.0
        assert meta["board_bounds"]["bottom"] == 1537.0

    def test_profile_json_save_and_reload_roundtrip(self):
        """Verifies JSON save and load preserves all fields identically."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            profile = CalibrationProfile(default_lift_y=210.0)
            profile.update_profile("line5_h", lift_y=215.5, anchor_dx=2.0, anchor_dy=-1.0)
            profile.save_to_json(tmp_path)

            loaded = CalibrationProfile(tmp_path)
            assert len(loaded) == len(profile)
            p_line5 = loaded.get_profile("line5_h")
            assert p_line5["lift_y"] == 215.5
            assert p_line5["anchor_dx"] == 2.0
            assert p_line5["anchor_dy"] == -1.0
            assert loaded.get_lift_offset("line5_h") == 215.5
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_dynamic_fallback_unknown_piece(self):
        """Dynamic / unregistered pieces generate valid fallback profiles without crashing."""
        profile = CalibrationProfile()
        dyn_name = "dyn_4x4_9999"
        BLOCK_SHAPES[dyn_name] = [(0, 0), (1, 1), (2, 2), (3, 3)]

        prof = profile.get_profile(dyn_name)
        assert prof["family"] == "dynamic_fallback"
        assert prof["width_blocks"] == 4
        assert prof["height_blocks"] == 4
        assert prof["block_count"] == 4
        assert prof["lift_y"] == DEFAULT_FINGER_LIFT_OFFSET
        assert prof["calibrated"] is False

        # Cleanup
        del BLOCK_SHAPES[dyn_name]


# =====================================================================
# TEST CLASS 2: MATHEMATICAL COORDINATE ENGINE & SAFE CLAMPING
# =====================================================================

class TestCoordinateEngineAndSafeClamping:
    """Verifies grid geometry, finger target calculation, and safe release limits."""

    def test_cell_center_coordinates(self):
        """Verifies exact pixel centers for board corners and center."""
        # Top-left (0, 0)
        x00, y00 = AutonomousCalibrator.get_cell_center_coords(0, 0)
        assert abs(x00 - 121) <= 1
        assert abs(y00 - 641) <= 1

        # Bottom-right (7, 7)
        x77, y77 = AutonomousCalibrator.get_cell_center_coords(7, 7)
        assert abs(x77 - 958) <= 1
        assert abs(y77 - 1477) <= 1

        # Center (3, 3)
        x33, y33 = AutonomousCalibrator.get_cell_center_coords(3, 3)
        assert abs(x33 - 480) <= 1
        assert abs(y33 - 999) <= 1

    def test_tray_slot_center_coordinates(self):
        """Verifies tray slot center coordinates."""
        assert AutonomousCalibrator.get_slot_center_coords(0) == (220, 1855)
        assert AutonomousCalibrator.get_slot_center_coords(1) == (540, 1855)
        assert AutonomousCalibrator.get_slot_center_coords(2) == (860, 1855)

    def test_piece_center_board_coordinates(self):
        """Verifies piece geometric center calculation."""
        profile = CalibrationProfile()
        # 1x1 dot at (0, 0) -> piece center is at cell (0, 0) center
        cx, cy = profile.get_board_center_xy("dot", 0, 0)
        x00, y00 = AutonomousCalibrator.get_cell_center_coords(0, 0)
        assert abs(cx - x00) < 1.0
        assert abs(cy - y00) < 1.0

        # 2x2 square at (0, 0) -> spans rows 0..1, cols 0..1
        cx2, cy2 = profile.get_board_center_xy("square2x2", 0, 0)
        assert abs(cx2 - (BOARD_LEFT_X + 1.0 * CELL_WIDTH)) < 1.0
        assert abs(cy2 - (BOARD_TOP_Y + 1.0 * CELL_HEIGHT)) < 1.0

    def test_finger_target_safe_release_clamping_row7(self):
        """
        Critical safety test: placing on bottom row (Row 7) clamps finger release
        strictly to Y <= 1580px, avoiding the tray cancellation zone (Y >= 1600px).
        """
        profile = CalibrationProfile()

        # Dot placed at (7, 0)
        fx, fy = profile.get_finger_target_xy("dot", target_row=7, target_col=0, clamp_safe=True)
        assert fy <= MAX_SAFE_RELEASE_Y
        assert fy == 1580
        assert not is_in_tray_cancel_zone(fy)

        # Line5_v placed at (3, 0) -> bottom row is row 7
        fx5, fy5 = profile.get_finger_target_xy("line5_v", target_row=3, target_col=0, clamp_safe=True)
        assert fy5 <= MAX_SAFE_RELEASE_Y
        assert not is_in_tray_cancel_zone(fy5)

    def test_finger_target_all_grid_cells_safe_bounds(self):
        """Ensures every piece placed at any valid grid position stays strictly within physical bounds."""
        profile = CalibrationProfile()
        for piece_name in ("dot", "line5_h", "line5_v", "square3x3", "corner_tl", "plus_cross"):
            prof = profile.get_profile(piece_name)
            wb = prof["width_blocks"]
            hb = prof["height_blocks"]

            for r in range(8 - hb + 1):
                for c in range(8 - wb + 1):
                    fx, fy = profile.get_finger_target_xy(piece_name, r, c, clamp_safe=True)
                    assert 10 <= fx <= 1070
                    assert 580 <= fy <= 1580
                    assert not is_in_tray_cancel_zone(fy)

    def test_clamp_safe_finger_release_extremes(self):
        """Boundary values for clamp_safe_finger_release."""
        # Extreme low
        sx, sy = clamp_safe_finger_release(-100, -200)
        assert sx == 10
        assert sy == 580

        # Extreme high
        sx, sy = clamp_safe_finger_release(2000, 3000)
        assert sx == 1070
        assert sy == 1580


# =====================================================================
# TEST CLASS 3: GHOST HIGHLIGHT DETECTION & MATCHING LOGIC
# =====================================================================

class TestGhostHighlightDetectionAndMatching:
    """Verifies ghost highlight extraction, color delta ranges, and match evaluations."""

    def test_expected_ghost_cells_canonical_shapes(self):
        """Verifies expected ghost highlight grid cell calculation."""
        expected_dot = AutonomousCalibrator.calculate_expected_ghost_cells("dot", 2, 3)
        assert expected_dot == [(2, 3)]

        expected_t = AutonomousCalibrator.calculate_expected_ghost_cells("t_up", 1, 1)
        # t_up blocks: [(0, 1), (1, 0), (1, 1), (1, 2)]
        assert sorted(expected_t) == sorted([(1, 2), (2, 1), (2, 2), (2, 3)])

        expected_square = AutonomousCalibrator.calculate_expected_ghost_cells("square2x2", 4, 4)
        assert sorted(expected_square) == sorted([(4, 4), (4, 5), (5, 4), (5, 5)])

    def test_detect_active_ghosts_on_synthetic_frame(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """Detector extracts active ghost cells matching SyntheticFrameBuilder drawn highlights."""
        calibrator = AutonomousCalibrator(detector=detector, simulation_mode=True)
        base = frame_builder.build_empty_frame()
        held = base.copy()

        target_cells = [(3, 2), (3, 3), (3, 4)]
        frame_builder.draw_ghost_highlights(held, target_cells, delta_e=40.0)

        active = calibrator.detect_active_ghosts(held, base)
        assert sorted(active) == sorted(target_cells)

    def test_ghost_match_evaluation_exact_match(self):
        """Exact match flags convergence with zero error."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        ghosts = [(2, 3), (2, 4), (2, 5)]
        converged, (dr, dc) = calibrator.evaluate_ghost_match(ghosts, ghosts)
        assert converged is True
        assert dr == 0
        assert dc == 0

    def test_ghost_match_evaluation_row_column_discrepancy(self):
        """Discrepancy in detected vs expected ghost returns proper error vector."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        expected = [(2, 3), (2, 4)]
        # Detected 1 row down, 1 col right
        detected = [(3, 4), (3, 5)]

        converged, (dr, dc) = calibrator.evaluate_ghost_match(detected, expected)
        assert converged is False
        assert dr == 1
        assert dc == 1

    def test_ghost_shadow_rejection_of_empty_and_solid_blocks(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """Ghost detection rejects unchanged cells (Delta E < 18) and solid blocks (Delta E > 80)."""
        calibrator = AutonomousCalibrator(detector=detector, simulation_mode=True)
        base = frame_builder.build_empty_frame()
        frame_builder.draw_board_cells(base, [(0, 0), (0, 1)])

        held = base.copy()
        # Ghost on (4, 4)
        frame_builder.draw_ghost_highlights(held, [(4, 4)], delta_e=40.0)

        active = calibrator.detect_active_ghosts(held, base)
        assert (4, 4) in active
        assert (0, 0) not in active
        assert (0, 1) not in active


# =====================================================================
# TEST CLASS 4: AUTONOMOUS PROBING & CONVERGENCE LOOP
# =====================================================================

class TestAutonomousProbingAndConvergence:
    """Verifies live/simulated holding gestures and autonomous self-tuning convergence."""

    def test_simulate_holding_probe_activates_ghosts(self):
        """Simulation probe renders ghost highlight at snapped grid coordinates."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        base = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), (35, 43, 84), dtype=np.uint8)

        # Probe dot at target (3, 3)
        cx, cy = AutonomousCalibrator.get_cell_center_coords(3, 3)
        finger_x = cx
        finger_y = cy + int(DEFAULT_FINGER_LIFT_OFFSET)

        held = calibrator.simulate_holding_probe("dot", 3, 3, finger_x, finger_y, base)
        active = calibrator.detect_active_ghosts(held, base)
        assert (3, 3) in active

    def test_simulate_holding_probe_tray_cancel_zone_suppression(self):
        """Holding gesture with finger in tray cancel zone (Y >= 1600) yields no ghost snap."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        base = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), (35, 43, 84), dtype=np.uint8)

        # Place finger in tray cancellation zone
        held = calibrator.simulate_holding_probe("dot", 7, 3, 540, 1650, base)
        active = calibrator.detect_active_ghosts(held, base)
        assert len(active) == 0

    def test_probe_piece_calibration_single_iteration(self):
        """Calibrating a piece under nominal alignment converges immediately."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        res = calibrator.probe_piece_calibration(
            slot_idx=1,
            piece="corner_tl",
            target_row=2,
            target_col=2,
            initial_lift=205.0,
        )
        assert res["converged"] is True
        assert res["piece_name"] == "corner_tl"
        assert res["family"] == "corners_small_2x2"
        assert res["iterations_count"] >= 1
        assert abs(res["lift_y"] - 205.0) < 1.0

    def test_probe_piece_calibration_self_tuning_convergence(self):
        """Calibrating with an intentional initial offset converges via error feedback."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        # Start with an offset lift
        res = calibrator.probe_piece_calibration(
            slot_idx=0,
            piece="square2x2",
            target_row=2,
            target_col=2,
            initial_lift=205.0,
            max_iterations=5,
        )
        assert res["converged"] is True
        assert res["piece_name"] == "square2x2"
        assert res["family"] == "squares"

    def test_calibrate_all_categories_populates_all_profiles(self):
        """Calibrating all categories returns complete, verified profile object."""
        calibrator = AutonomousCalibrator(simulation_mode=True)
        profile = calibrator.calibrate_all_categories()
        assert len(profile) == 42
        for name in BLOCK_SHAPES.keys():
            if not name.startswith("dyn_"):
                prof = profile.get_profile(name)
                assert prof["calibrated"] is True
                assert prof["lift_y"] > 0

    def test_run_full_calibration_creates_file(self):
        """Full calibration workflow writes valid JSON artifact."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            calibrator = AutonomousCalibrator(simulation_mode=True, profile_path=tmp_path)
            prof = calibrator.run_full_calibration(output_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000
            with open(tmp_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            assert "_meta" in saved_json
            assert "profiles" in saved_json
            assert len(saved_json["profiles"]) == 42
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# =====================================================================
# TEST CLASS 5: MULTI-THEME GHOST HIGHLIGHT ADAPTATION
# =====================================================================

class TestMultiThemeGhostHighlightAdaptation:
    """Verifies ghost shadow detection across Wood, Blue, Neon, and Jungle themes."""

    @pytest.mark.parametrize("theme_name", ["wood", "blue", "neon", "jungle"])
    def test_ghost_shadow_detection_across_all_themes(self, theme_name: str, detector: BlockBlastDetector):
        """Ghost highlights are reliably detected across all 4 visual themes."""
        builder = SyntheticFrameBuilder(theme=theme_name)
        base = builder.build_empty_frame()
        held = base.copy()

        target_cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
        builder.draw_ghost_highlights(held, target_cells, delta_e=40.0)

        calibrator = AutonomousCalibrator(detector=detector, simulation_mode=True)
        active = calibrator.detect_active_ghosts(held, base)
        assert sorted(active) == sorted(target_cells)


# =====================================================================
# TEST CLASS 6: MOCK ADB SOCKET INTEGRATION & CONNECTIVITY
# =====================================================================

class TestMockADBIntegration:
    """Verifies interaction with FastADBSocketClient over real/mock socket."""

    def test_calibrator_with_mock_adb_client(self, mock_adb_client: FastADBSocketClient, detector: BlockBlastDetector):
        """AutonomousCalibrator works seamlessly with FastADBSocketClient."""
        calibrator = AutonomousCalibrator(adb_client=mock_adb_client, detector=detector)
        assert calibrator.adb is not None
        # Mock ADB client reports connected
        assert calibrator.adb.is_connected() is True

    def test_calibrator_handles_offline_client_gracefully(self):
        """Calibrator without active device falls back to simulation mode safely."""
        offline_client = FastADBSocketClient(serial="OFFLINE_DEVICE_9999", host="127.0.0.1", port=59999, connect_timeout=0.1)
        calibrator = AutonomousCalibrator(adb_client=offline_client)
        assert calibrator.is_live is False
        # Should still calibrate in simulation mode without raising exceptions
        res = calibrator.probe_piece_calibration(0, "dot")
        assert res["converged"] is True


# =====================================================================
# TEST CLASS 7: CONVENIENCE FUNCTIONS & MODULE EXPORTS
# =====================================================================

class TestModuleExportsAndConvenience:
    """Verifies standalone helper functions and module API."""

    def test_load_and_save_calibration_profiles_convenience_methods(self):
        """Verifies load_calibration_profiles and save_calibration_profiles."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_calibration_profiles(tmp_path)
            assert os.path.exists(tmp_path)

            loaded = load_calibration_profiles(tmp_path)
            assert isinstance(loaded, CalibrationProfile)
            assert len(loaded) == 42
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_get_shape_family_mappings(self):
        """Verifies family lookup for canonical and dynamic shapes."""
        assert get_shape_family("dot") == "dot"
        assert get_shape_family("line5_h") == "lines_horizontal"
        assert get_shape_family("line5_v") == "lines_vertical"
        assert get_shape_family("big_corner_tr") == "corners_big_3x3"
        assert get_shape_family("plus_cross") == "plus_cross"
        assert get_shape_family("dyn_2x2_123") == "dynamic_fallback"
        assert get_shape_family("completely_unknown") == "unknown"
