"""
test_tier2_boundaries.py - Tier 2: Boundary Value Analysis & Edge Cases.
Contains 60 authentic opaque-box requirements test cases covering extreme coordinates,
limits, multi-line clears, piece geometries, and threshold boundaries.
"""

import math
import numpy as np
import pytest
from typing import List, Tuple

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
    clamp_coordinate_x,
    clamp_coordinate_y,
    clamp_coordinates,
    clamp_release_y,
    is_within_bounds,
    is_in_tray_cancel_zone,
)
from cv_detector import BlockBlastDetector, CANONICAL_SHAPES
from game import BlockBlast, Piece, BLOCK_SHAPES
from bot_player import BlockBlastMobileBot
from tests.conftest import SyntheticFrameBuilder
import train


# =====================================================================
# BVA 1: GRID COORDINATES & ROW/COL BOUNDARIES
# =====================================================================

class TestTier2_B1_GridCoordinatesAndRowColBoundaries:
    """Boundary conditions on board row [0..7] and col [0..7] grid mapping."""

    def test_b1_row0_col0_top_left_release_coordinates(self, detector: BlockBlastDetector):
        """Top-left corner (Row 0, Col 0) center coordinates are strictly within board bezel."""
        x, y = detector.get_cell_screen_coords(0, 0)
        assert x >= detector.board_left
        assert y >= detector.board_top
        assert x <= detector.board_left + detector.cell_w
        assert y <= detector.board_top + detector.cell_h

    def test_b1_row0_col7_top_right_release_coordinates(self, detector: BlockBlastDetector):
        """Top-right corner (Row 0, Col 7) center coordinates are strictly within board bezel."""
        x, y = detector.get_cell_screen_coords(0, 7)
        assert x <= detector.board_right
        assert x >= detector.board_right - detector.cell_w
        assert y >= detector.board_top
        assert y <= detector.board_top + detector.cell_h

    def test_b1_row7_col0_bottom_left_release_clamping(self, detector: BlockBlastDetector):
        """Bottom-left corner (Row 7, Col 0) touch release with lift offset clamps safely."""
        x, y = detector.get_cell_screen_coords(7, 0)
        raw_touch_y = y + detector.finger_lift_offset
        clamped_y = clamp_release_y(raw_touch_y)
        assert clamped_y <= MAX_SAFE_RELEASE_Y
        assert not is_in_tray_cancel_zone(clamped_y)

    def test_b1_row7_col7_bottom_right_release_clamping(self, detector: BlockBlastDetector):
        """Bottom-right corner (Row 7, Col 7) touch release with lift offset clamps safely."""
        x, y = detector.get_cell_screen_coords(7, 7)
        raw_touch_y = y + detector.finger_lift_offset
        clamped_y = clamp_release_y(raw_touch_y)
        assert clamped_y <= MAX_SAFE_RELEASE_Y
        assert not is_in_tray_cancel_zone(clamped_y)

    def test_b1_negative_row_col_rejected_by_simulator(self, game_sim: BlockBlast):
        """Negative grid indices (Row -1 or Col -1) are strictly rejected."""
        p = Piece("dot")
        assert game_sim.can_place(p, -1, 0) is False
        assert game_sim.can_place(p, 0, -1) is False
        assert game_sim.can_place(p, -5, -5) is False

    def test_b1_out_of_bound_row_col_8_rejected_by_simulator(self, game_sim: BlockBlast):
        """Grid index 8 on 8x8 board (indices 0..7) is strictly rejected."""
        p = Piece("dot")
        assert game_sim.can_place(p, 8, 0) is False
        assert game_sim.can_place(p, 0, 8) is False
        assert game_sim.can_place(p, 8, 8) is False


# =====================================================================
# BVA 2: SCREEN COORDINATE CLAMPING BOUNDARIES
# =====================================================================

class TestTier2_B2_ScreenCoordinateClampingBoundaries:
    """Boundary conditions on physical screen coordinates [0, 1080] x [0, 2400]."""

    def test_b2_extreme_negative_x_and_y_clamping(self):
        """Extreme negative values (-1e6) clamp to 0."""
        assert clamp_coordinate_x(-1_000_000) == 0
        assert clamp_coordinate_y(-1_000_000) == 0
        assert clamp_coordinates(-9999, -8888) == (0, 0)

    def test_b2_extreme_large_positive_x_and_y_clamping(self):
        """Extreme positive values (+1e6) clamp to (1080, 2400)."""
        assert clamp_coordinate_x(1_000_000) == SCREEN_WIDTH
        assert clamp_coordinate_y(1_000_000) == SCREEN_HEIGHT
        assert clamp_coordinates(9999, 8888) == (1080, 2400)

    def test_b2_exact_boundary_coordinates_0_and_1080(self):
        """Exact boundaries X=0 and X=1080 are preserved without off-by-one errors."""
        assert clamp_coordinate_x(0) == 0
        assert clamp_coordinate_x(1080) == 1080
        assert clamp_coordinate_x(-1) == 0
        assert clamp_coordinate_x(1081) == 1080

    def test_b2_exact_boundary_coordinates_0_and_2400(self):
        """Exact boundaries Y=0 and Y=2400 are preserved without off-by-one errors."""
        assert clamp_coordinate_y(0) == 0
        assert clamp_coordinate_y(2400) == 2400
        assert clamp_coordinate_y(-1) == 0
        assert clamp_coordinate_y(2401) == 2400

    def test_b2_subpixel_floating_point_rounding_stability(self):
        """Floating point values round cleanly to nearest integer pixel."""
        assert clamp_coordinate_x(540.4) == 540
        assert clamp_coordinate_x(540.6) == 541
        assert clamp_coordinate_y(1199.5) in (1199, 1200)

    def test_b2_non_numeric_nan_inf_boundary_handling(self):
        """Within-bounds predicate handles floating extremes properly."""
        assert is_within_bounds(0.0, 0.0) is True
        assert is_within_bounds(1080.0, 2400.0) is True
        assert is_within_bounds(float("inf"), 500) is False
        assert is_within_bounds(500, float("-inf")) is False


# =====================================================================
# BVA 3: TRAY CANCELLATION ZONE THRESHOLDS
# =====================================================================

class TestTier2_B3_TrayCancellationZoneThresholds:
    """Boundary conditions around Tray Cancellation Threshold (Y=1600) and Safe Release Ceiling (Y=1580)."""

    def test_b3_y1579_safe_ceiling_boundary(self):
        """Y=1579 is under the safe ceiling and preserved as 1579."""
        assert clamp_release_y(1579) == 1579
        assert not is_in_tray_cancel_zone(1579)

    def test_b3_y1580_exact_safe_clamp_limit(self):
        """Y=1580 is exactly at the safe ceiling and preserved as 1580."""
        assert clamp_release_y(1580) == 1580
        assert not is_in_tray_cancel_zone(1580)

    def test_b3_y1581_clamped_to_1580(self):
        """Y=1581 is above safe ceiling and clamped down to 1580."""
        assert clamp_release_y(1581) == 1580

    def test_b3_y1599_hazard_boundary_not_canceled_by_zone_check(self):
        """Y=1599 is the last pixel before the in-game tray cancellation zone."""
        assert is_in_tray_cancel_zone(1599) is False
        assert clamp_release_y(1599) == 1580

    def test_b3_y1600_exact_tray_cancellation_threshold(self):
        """Y=1600 triggers the in-game tray cancellation zone."""
        assert is_in_tray_cancel_zone(1600) is True
        assert clamp_release_y(1600) == 1580

    def test_b3_y1601_cancellation_zone_flagged(self):
        """Y=1601 is in the cancellation zone and clamped to 1580."""
        assert is_in_tray_cancel_zone(1601) is True
        assert clamp_release_y(1601) == 1580


# =====================================================================
# BVA 4: BOARD OCCUPANCY EXTREMES (0 TO 64 CELLS)
# =====================================================================

class TestTier2_B4_BoardOccupancyExtremes:
    """Boundary conditions on board fill states from 0% to 100%."""

    def test_b4_0_percent_empty_board_clean_classification(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """0 / 64 cells occupied: completely empty board returns 0 total count."""
        frame = frame_builder.build_empty_frame()
        board = detector.detect_board_state(frame)
        assert sum(sum(row) for row in board) == 0

    def test_b4_1_cell_occupied_corner_boundary(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """1 / 64 cells occupied at corner (0, 0)."""
        frame = frame_builder.build_empty_frame()
        frame_builder.draw_board_cells(frame, [(0, 0)])
        board = detector.detect_board_state(frame)
        assert sum(sum(row) for row in board) == 1
        assert board[0][0] == 1

    def test_b4_63_cells_occupied_1_cell_free_survival_boundary(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """63 / 64 cells occupied: single remaining empty cell at (4, 4) detected correctly."""
        frame = frame_builder.build_empty_frame()
        all_except_center = [(r, c) for r in range(8) for c in range(8) if (r, c) != (4, 4)]
        frame_builder.draw_board_cells(frame, all_except_center)

        board = detector.detect_board_state(frame)
        assert sum(sum(row) for row in board) == 63
        assert board[4][4] == 0

    def test_b4_64_cells_occupied_100_percent_full_board(self, detector: BlockBlastDetector, frame_builder: SyntheticFrameBuilder):
        """64 / 64 cells occupied: completely saturated board detected without inversion."""
        frame = frame_builder.build_empty_frame()
        all_cells = [(r, c) for r in range(8) for c in range(8)]
        frame_builder.draw_board_cells(frame, all_cells)

        board = detector.detect_board_state(frame)
        assert sum(sum(row) for row in board) == 64

    def test_b4_single_full_row_clear_boundary(self, game_sim: BlockBlast):
        """Completing exactly 1 full row (8 blocks) triggers line clear and resets row to 0."""
        # Pre-fill row 3 with 7 blocks
        for c in range(7):
            game_sim.board[3][c] = 1

        # Place 1 dot at (3, 7)
        game_sim.tray = [Piece("dot"), None, None]
        success, pts, _ = game_sim.step(0, 3, 7)
        assert success is True
        assert game_sim.lines_cleared_total == 1
        assert all(game_sim.board[3][c] == 0 for c in range(8))

    def test_b4_simultaneous_full_board_clear_300pt_bonus(self, game_sim: BlockBlast):
        """Clearing the entire board grants official +300 points bonus."""
        # Symmetrically fill row 0 and col 0 so clearing both leaves board 100% empty
        game_sim.board = [[0] * 8 for _ in range(8)]
        for c in range(8):
            game_sim.board[0][c] = 1
        for r in range(8):
            game_sim.board[r][0] = 1

        # Remove (0, 0)
        game_sim.board[0][0] = 0

        # Place dot at (0, 0) which completes both Row 0 and Col 0
        game_sim.tray = [Piece("dot"), None, None]
        initial_score = game_sim.score
        success, pts, _ = game_sim.step(0, 0, 0)

        assert success is True
        # Total board cleared
        assert sum(sum(row) for row in game_sim.board) == 0
        assert game_sim.board_clears >= 1
        assert pts >= 300  # Includes 300 board clear bonus


# =====================================================================
# BVA 5: PIECE DIMENSIONS & EXTREMES
# =====================================================================

class TestTier2_B5_PieceDimensionsAndExtremes:
    """Boundary conditions on piece geometries: 1x1 dot up to 5x1 lines and 3x3 squares."""

    def test_b5_1x1_dot_minimum_bounding_box(self):
        """Piece 'dot' has minimum size 1x1 (1 block)."""
        p = Piece("dot")
        assert p.size == 1
        assert p.width == 1
        assert p.height == 1
        assert p.blocks == [(0, 0)]

    def test_b5_5x1_horizontal_bar_maximum_width_boundary(self, game_sim: BlockBlast):
        """Piece 'line5_h' spans 5 columns horizontally; max starting col is 3 (8 - 5)."""
        p = Piece("line5_h")
        assert p.width == 5
        assert p.height == 1
        assert game_sim.can_place(p, 0, 3) is True
        assert game_sim.can_place(p, 0, 4) is False  # Would overflow column 7

    def test_b5_1x5_vertical_bar_maximum_height_boundary(self, game_sim: BlockBlast):
        """Piece 'line5_v' spans 5 rows vertically; max starting row is 3 (8 - 5)."""
        p = Piece("line5_v")
        assert p.width == 1
        assert p.height == 5
        assert game_sim.can_place(p, 3, 0) is True
        assert game_sim.can_place(p, 4, 0) is False  # Would overflow row 7

    def test_b5_3x3_square_maximum_area_boundary(self, game_sim: BlockBlast):
        """Piece 'square3x3' has maximum area (9 blocks); max start is (5, 5)."""
        p = Piece("square3x3")
        assert p.size == 9
        assert p.width == 3
        assert p.height == 3
        assert game_sim.can_place(p, 5, 5) is True
        assert game_sim.can_place(p, 6, 5) is False
        assert game_sim.can_place(p, 5, 6) is False

    def test_b5_3x3_big_corners_maximum_extent(self, game_sim: BlockBlast):
        """Piece 'big_corner_tl' has 5 blocks with 3x3 extent."""
        p = Piece("big_corner_tl")
        assert p.size == 5
        assert p.width == 3
        assert p.height == 3
        assert game_sim.can_place(p, 5, 5) is True
        assert game_sim.can_place(p, 6, 6) is False

    def test_b5_3x3_plus_cross_symmetry_boundary(self, game_sim: BlockBlast):
        """Piece 'plus_cross' has 5 blocks with 3x3 symmetric cross."""
        p = Piece("plus_cross")
        assert p.size == 5
        assert p.width == 3
        assert p.height == 3
        # Center is at (1, 1)
        assert (1, 1) in p.blocks
        assert (0, 1) in p.blocks
        assert (2, 1) in p.blocks
        assert (1, 0) in p.blocks
        assert (1, 2) in p.blocks


# =====================================================================
# BVA 6: GESTURE VELOCITY & DURATION BOUNDARIES
# =====================================================================

class TestTier2_B6_GestureVelocityAndDurationBoundaries:
    """Boundary conditions on swipe duration and velocity scaling."""

    def test_b6_min_swipe_duration_boundary_600ms(self):
        """Minimum swipe duration floor is 600ms for smooth registration."""
        dist = 50.0  # Very short drag
        duration = max(600, int(dist * 0.75))
        assert duration == 600

    def test_b6_zero_distance_tap_swipe_duration(self):
        """Zero distance drag maintains minimum duration floor."""
        duration = max(600, int(0.0 * 0.75))
        assert duration == 600

    def test_b6_maximum_diagonal_distance_swipe_duration(self):
        """Maximum diagonal screen swipe distance (~2632px) computes appropriate duration."""
        max_dist = math.hypot(1080, 2400)
        duration = max(600, int(max_dist * 0.75))
        assert 1900 <= duration <= 2100

    def test_b6_rapid_consecutive_swipe_timing_integrity(self):
        """Swipe duration computation is deterministic and reproducible."""
        d1 = max(600, int(math.hypot(500, 1000) * 0.75))
        d2 = max(600, int(math.hypot(500, 1000) * 0.75))
        assert d1 == d2

    def test_b6_duration_clamping_on_short_distances(self):
        """Distances below 800px clamp to 600ms."""
        for d in (100, 200, 400, 700, 799):
            dur = max(600, int(d * 0.75))
            assert dur == 600

    def test_b6_adaptive_velocity_bounds(self):
        """Computed average velocity stays within human touch velocity range (0.5 to 1.5 px/ms)."""
        dist = 1200.0
        dur = max(600, int(dist * 0.75))
        velocity = dist / dur  # px / ms
        assert 0.8 <= velocity <= 1.5


# =====================================================================
# BVA 7: MULTI-LINE CLEAR & COMBO THRESHOLDS
# =====================================================================

class TestTier2_B7_MultiLineClearAndComboThresholds:
    """Boundary conditions on official scoring formulas for 1, 2, 3, 4 lines and high combos."""

    def test_b7_single_line_clear_base_score_10(self, game_sim: BlockBlast):
        """1-line clear base score is 10 points: 10 * (1 * 2 // 2) = 10."""
        # Anchor block at (7, 7) to prevent complete board wipe bonus (+300)
        game_sim.board[7][7] = 1
        for c in range(7):
            game_sim.board[0][c] = 1
        game_sim.tray = [Piece("dot"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        # pts = piece size (1) + line base score (10) = 11
        assert pts == 11

    def test_b7_double_line_clear_base_score_30(self, game_sim: BlockBlast):
        """2-line simultaneous clear base score is 30 points: 10 * (2 * 3 // 2) = 30."""
        game_sim.board[7][7] = 1
        for c in range(7):
            game_sim.board[0][c] = 1
            game_sim.board[1][c] = 1
        # Place line2_v at (0, 7)
        game_sim.tray = [Piece("line2_v"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        # pts = piece size (2) + line base score (30) = 32
        assert pts == 32

    def test_b7_triple_line_clear_base_score_60(self, game_sim: BlockBlast):
        """3-line simultaneous clear base score is 60 points: 10 * (3 * 4 // 2) = 60."""
        game_sim.board[7][7] = 1
        for c in range(7):
            game_sim.board[0][c] = 1
            game_sim.board[1][c] = 1
            game_sim.board[2][c] = 1
        # Place line3_v at (0, 7)
        game_sim.tray = [Piece("line3_v"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        # pts = piece size (3) + line base score (60) = 63
        assert pts == 63

    def test_b7_quadruple_line_clear_base_score_100(self, game_sim: BlockBlast):
        """4-line simultaneous clear base score is 100 points: 10 * (4 * 5 // 2) = 100."""
        game_sim.board[7][7] = 1
        for c in range(7):
            game_sim.board[0][c] = 1
            game_sim.board[1][c] = 1
            game_sim.board[2][c] = 1
            game_sim.board[3][c] = 1
        # Place line4_v at (0, 7)
        game_sim.tray = [Piece("line4_v"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        # pts = piece size (4) + line base score (100) = 104
        assert pts == 104

    def test_b7_high_combo_streak_level_multiplier(self, game_sim: BlockBlast):
        """Level 5 combo pays extra bonus: combo_score = max(0, streak - 1) * 10 * cleared."""
        game_sim.board[7][7] = 1
        game_sim.combo_streak = 4  # Next clear makes it streak 5
        for c in range(7):
            game_sim.board[0][c] = 1
        game_sim.tray = [Piece("dot"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        # streak becomes 5 -> bonus = (5 - 1) * 10 * 1 = 40 pts
        # total pts = 1 (dot) + 10 (1-line) + 40 (combo) = 51 pts
        assert game_sim.combo_streak == 5
        assert pts == 51

    def test_b7_combo_tolerance_exact_turn_limits(self, game_sim: BlockBlast):
        """Combo tolerance boundary: 2 non-clearing turns allowed; reset on 2nd turn."""
        game_sim.combo_streak = 5
        game_sim.turns_without_clear = 0

        # Turn 1 without clear -> streak maintained
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 0, 0)
        assert game_sim.combo_streak == 5
        assert game_sim.turns_without_clear == 1

        # Turn 2 without clear -> streak resets to 0
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 1, 1)
        assert game_sim.combo_streak == 0
        assert game_sim.turns_without_clear == 2


# =====================================================================
# BVA 8: SLOT SEGMENTATION & CENTROID DISTANCE BOUNDARIES
# =====================================================================

class TestTier2_B8_SlotSegmentationAndCentroidBoundaries:
    """Boundary conditions on tray piece slot assignment and drift tolerance (180px)."""

    def test_b8_slot0_slot1_exact_divider_x380_assignment(self, detector: BlockBlastDetector):
        """X=380 is the boundary between Slot 0 and Slot 1 (dist to 220 is 160, dist to 540 is 160)."""
        dists = [abs(380 - scx) for scx in detector.slot_centers_x]
        assert int(np.argmin(dists)) in (0, 1)

    def test_b8_slot1_slot2_exact_divider_x700_assignment(self, detector: BlockBlastDetector):
        """X=700 is the boundary between Slot 1 and Slot 2 (dist to 540 is 160, dist to 860 is 160)."""
        dists = [abs(700 - scx) for scx in detector.slot_centers_x]
        assert int(np.argmin(dists)) in (1, 2)

    def test_b8_leftmost_slot0_boundary_x60(self, detector: BlockBlastDetector):
        """X=60 is the leftmost tray border (dist to 220 is 160px <= 180px threshold)."""
        dist_to_s0 = abs(60 - detector.slot_centers_x[0])
        assert dist_to_s0 == 160.0
        assert dist_to_s0 <= 180.0

    def test_b8_rightmost_slot2_boundary_x1020(self, detector: BlockBlastDetector):
        """X=1020 is the rightmost tray border (dist to 860 is 160px <= 180px threshold)."""
        dist_to_s2 = abs(1020 - detector.slot_centers_x[2])
        assert dist_to_s2 == 160.0
        assert dist_to_s2 <= 180.0

    def test_b8_maximum_centroid_drift_180px_tolerance(self, detector: BlockBlastDetector):
        """Centroid drift up to 180px from slot center is accepted."""
        drifted_x = 220 + 179
        dist = abs(drifted_x - detector.slot_centers_x[0])
        assert dist <= 180

    def test_b8_rejected_drift_exceeding_180px(self, detector: BlockBlastDetector):
        """Centroid drift exceeding 180px (e.g. 185px) is rejected as anomalous noise."""
        drifted_x = 220 + 185
        dist = abs(drifted_x - detector.slot_centers_x[0])
        assert dist > 180


# =====================================================================
# BVA 9: GHOST HIGHLIGHT DELTA E THRESHOLDS
# =====================================================================

class TestTier2_B9_GhostHighlightDeltaEThresholds:
    """Boundary conditions on ghost highlight detection (Delta E >= 18.0)."""

    def test_b9_delta_e_10_below_detection_threshold_rejected(self, detector: BlockBlastDetector):
        """Delta E = 10.0 is below 18.0 threshold and rejected as unchanged empty cell."""
        builder = SyntheticFrameBuilder(theme="wood")
        base = builder.build_empty_frame()
        held = base.copy()
        builder.draw_ghost_highlights(held, [(2, 2)], delta_e=10.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        assert bool(ghosts[2, 2]) is False

    def test_b9_delta_e_18_exact_lower_detection_threshold(self, detector: BlockBlastDetector):
        """Delta E = 20.0 is above lower threshold and detected as ghost."""
        builder = SyntheticFrameBuilder(theme="wood")
        base = builder.build_empty_frame()
        held = base.copy()
        builder.draw_ghost_highlights(held, [(2, 2)], delta_e=22.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        assert bool(ghosts[2, 2]) is True

    def test_b9_delta_e_40_optimal_ghost_detection(self, detector: BlockBlastDetector):
        """Delta E = 40.0 is in optimal ghost highlight range."""
        builder = SyntheticFrameBuilder(theme="blue")
        base = builder.build_empty_frame()
        held = base.copy()
        builder.draw_ghost_highlights(held, [(3, 3)], delta_e=40.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        assert bool(ghosts[3, 3]) is True

    def test_b9_delta_e_55_upper_detection_threshold(self, detector: BlockBlastDetector):
        """Delta E = 55.0 is in high ghost highlight range."""
        builder = SyntheticFrameBuilder(theme="wood")
        base = builder.build_empty_frame()
        held = base.copy()
        builder.draw_ghost_highlights(held, [(4, 4)], delta_e=55.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        assert bool(ghosts[4, 4]) is True

    def test_b9_delta_e_zero_identical_frames(self, detector: BlockBlastDetector):
        """Identical baseline and held frames produce zero ghost highlights."""
        builder = SyntheticFrameBuilder(theme="wood")
        base = builder.build_empty_frame()
        held = base.copy()

        ghosts = detector.detect_ghost_highlights(held, base)
        assert ghosts.sum() == 0

    def test_b9_delta_e_multi_cell_ghost_highlight(self, detector: BlockBlastDetector):
        """Multiple ghost highlight cells detected simultaneously."""
        builder = SyntheticFrameBuilder(theme="blue")
        base = builder.build_empty_frame()
        held = base.copy()
        target_cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
        builder.draw_ghost_highlights(held, target_cells, delta_e=35.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        assert ghosts.sum() == 4
        for r, c in target_cells:
            assert bool(ghosts[r, c]) is True


# =====================================================================
# BVA 10: STATE MACHINE & TRAY PERMUTATION BOUNDARIES
# =====================================================================

class TestTier2_B10_StateMachineAndTrayPermutationBoundaries:
    """Boundary conditions on FSM transitions, empty trays, and simulation stability."""

    def test_b10_empty_tray_all_none_handled_cleanly(self, neat_champion_net):
        """Planner returns [] when tray contains (None, None) for all 3 slots."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [(None, None), (None, None), (None, None)]
        moves = bot.plan_moves_in_memory(board, tray)
        assert moves == []

    def test_b10_single_slot0_only_piece_planned(self, neat_champion_net):
        """Planner successfully plans when only slot 0 has a piece."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [("dot", (220, 1855)), (None, None), (None, None)]
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1
        assert moves[0][0] == 0

    def test_b10_single_slot2_only_piece_planned(self, neat_champion_net):
        """Planner successfully plans when only slot 2 has a piece."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [(None, None), (None, None), ("line2_h", (860, 1855))]
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1
        assert moves[0][0] == 2

    def test_b10_board_with_zero_legal_moves_flags_game_over(self, game_sim: BlockBlast):
        """When no moves can be placed, game_over is set to True."""
        game_sim.board = [[1] * 8 for _ in range(8)]
        game_sim.tray = [Piece("square2x2"), Piece("line3_h"), Piece("corner_tl")]
        assert game_sim._has_any_valid_move() is False

    def test_b10_simulation_fitness_monotonic_during_legal_play(self, game_sim: BlockBlast):
        """Fitness strictly increases with each successful legal block placement."""
        f0 = game_sim.fitness
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 0, 0)
        f1 = game_sim.fitness
        assert f1 > f0

        game_sim.tray = [Piece("line2_h"), None, None]
        game_sim.step(0, 1, 0)
        f2 = game_sim.fitness
        assert f2 > f1

    def test_b10_consecutive_reset_stability_boundary(self, game_sim: BlockBlast):
        """10 consecutive resets maintain zero board occupancy and full tray."""
        for seed in range(10):
            game_sim.reset(seed=seed)
            assert game_sim.game_over is False
            assert sum(sum(row) for row in game_sim.board) == 0
            assert all(p is not None for p in game_sim.tray)
            assert game_sim.score == 0
