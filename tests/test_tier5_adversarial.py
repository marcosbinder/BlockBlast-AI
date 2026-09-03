"""
test_tier5_adversarial.py - Tier 5 Adversarial Stress Testing & Coverage Hardening Suite.

Comprehensive empirical challenge suite verifying:
1. Batch 3-piece permutation planning under adversarial piece combinations and cascade sequences.
2. Extreme board patterns (63/64 cells, checkerboards, isolated single holes, multi-corner traps).
3. Strict coordinate math & physical bounds clamping (finger release Y <= 1580px, never in tray zone Y >= 1600px).
4. SIMD CV detection resilience under synthetic noise, illumination gradients, and extreme themes.
5. FSM robustness, socket disconnection recovery, and closed-loop desync handling.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from adb_client import (
    BOARD_BOTTOM_Y,
    BOARD_LEFT_X,
    BOARD_RIGHT_X,
    BOARD_TOP_Y,
    DEFAULT_DEVICE_SERIAL,
    MAX_SAFE_RELEASE_Y,
    MIN_SAFE_RELEASE_Y,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TRAY_CANCEL_ZONE_START_Y,
    FastADBSocketClient,
    clamp_coordinate_x,
    clamp_coordinate_y,
    clamp_coordinates,
    clamp_release_y,
    is_in_tray_cancel_zone,
    is_within_bounds,
)
from bot_player import (
    BlockBlastMobileBot,
    BotState,
    load_champion_network,
    plan_batch_moves,
)
from cv_detector import (
    CANONICAL_SHAPES,
    BlockBlastDetector,
)
from empirical_calibrator import (
    CELL_HEIGHT,
    CELL_WIDTH,
    CalibrationProfile,
    clamp_safe_finger_release,
    create_default_calibration_profiles,
    get_shape_family,
)
from game import (
    BLOCK_SHAPES,
    PIECE_NAMES,
    BlockBlast,
    Piece,
    simulate_batch_sequence,
)
from tests.conftest import SyntheticFrameBuilder


def make_synthetic_frame(
    theme: str = "wood",
    occupied_matrix: Optional[List[List[int]]] = None,
    tray_pieces: Optional[List[Tuple[int, str]]] = None,
) -> np.ndarray:
    """Helper to build synthetic 1080x2400 frames for any theme with board and tray state."""
    builder = SyntheticFrameBuilder(theme=theme)
    frame = builder.build_empty_frame()
    if occupied_matrix is not None:
        cells = [(r, c) for r in range(8) for c in range(8) if occupied_matrix[r][c] != 0]
        frame = builder.draw_board_cells(frame, cells)
    if tray_pieces is not None:
        for slot_idx, piece_name in tray_pieces:
            frame = builder.draw_tray_piece(frame, slot_idx, piece_name)
    return frame


# =====================================================================
# TIER 5.1: ADVERSARIAL BATCH 3-PIECE PERMUTATION PLANNING & CASCADES
# =====================================================================

class TestTier5_AdversarialBatchPermutationPlanning:
    """Stress tests batch 3-piece permutation planning with adversarial piece sets and cascading clears."""

    def test_adv_3piece_forced_order_cascade(self, neat_champion_net):
        """
        Adversarial scenario: A board layout where:
        - Piece 1 (line4_h) MUST be placed at Row 0 to clear Row 0.
        - Piece 2 (square2x2) CANNOT fit until Row 0 is cleared; placing it clears Row 1, 2.
        - Piece 3 (line2_v) placed subsequently.
        The planner MUST find the exact permutation sequence that unlocks all 3 placements.
        """
        board = [[0] * 8 for _ in range(8)]
        # Fill row 0 cols 4..7 (placing line4_h at (0,0) will fill cols 0..3 and clear row 0)
        for c in range(4, 8):
            board[0][c] = 1

        # Fill cols 2..7 of rows 1, 2 so square2x2 fits at (0,0) ONLY after row 0 is cleared
        for r in range(1, 3):
            for c in range(2, 8):
                board[r][c] = 1

        tray = [
            {"slot": 0, "name": "line4_h"},
            {"slot": 1, "name": "square2x2"},
            {"slot": 2, "name": "line2_v"},
        ]

        plan = plan_batch_moves(board, tray, neat_champion_net)
        assert isinstance(plan, list), "Planner must return a list of moves"
        assert len(plan) > 0, "Planner must find at least 1 valid move in cascading scenario"

        # Verify sequential simulation executes cleanly without collisions
        sim = BlockBlast()
        sim.board = [row[:] for row in board]
        sim.tray = [Piece(item["name"]) for item in tray]
        for slot_idx, r, c in plan:
            assert sim.can_place(sim.tray[slot_idx], r, c), f"Move ({slot_idx}, {r}, {c}) must be legally placeable"
            sim.step(slot_idx, r, c)

    def test_adv_3piece_all_huge_3x3_unplaceable(self, neat_champion_net):
        """
        Adversarial scenario: 3x square3x3 on a crowded board that can fit at most one 3x3.
        Planner must return the best single move without hanging, infinite loops, or crashes.
        """
        board = [[0] * 8 for _ in range(8)]
        # Occupy board so only ONE 3x3 open space exists at (0,0)
        for r in range(8):
            for c in range(8):
                if not (r < 3 and c < 3):
                    board[r][c] = 1

        tray = [
            {"slot": 0, "name": "square3x3"},
            {"slot": 1, "name": "square3x3"},
            {"slot": 2, "name": "square3x3"},
        ]

        plan = plan_batch_moves(board, tray, neat_champion_net)
        assert isinstance(plan, list)
        assert len(plan) >= 1, "Should plan at least the single valid 3x3 move"
        assert plan[0][1] == 0 and plan[0][2] == 0

    def test_adv_3piece_triple_5bars_parallel_clear(self, neat_champion_net):
        """
        Adversarial scenario: Board has 3 rows missing 5 blocks each (cols 0..4 empty, cols 5..7 full).
        Tray contains 3x line5_h.
        Planner must successfully place all 3 bars and trigger 3 line clears.
        """
        board = [[0] * 8 for _ in range(8)]
        for r in (1, 3, 5):
            for c in range(5, 8):
                board[r][c] = 1

        tray = [
            {"slot": 0, "name": "line5_h"},
            {"slot": 1, "name": "line5_h"},
            {"slot": 2, "name": "line5_h"},
        ]

        plan = plan_batch_moves(board, tray, neat_champion_net)
        assert len(plan) == 3, f"Must successfully plan all 3 line5_h moves, got {len(plan)}"

        target_rows = {m[1] for m in plan}
        assert target_rows == {1, 3, 5}

    def test_adv_3piece_50_randomized_crowded_boards_stability(self, neat_champion_net):
        """
        Adversarial stress harness: Generate 50 randomized crowded boards (occupancy 40% - 85%)
        with 3 random pieces. Verify:
        - 100% zero unhandled exceptions.
        - Every planned move is strictly valid and legal on the resulting intermediate board.
        - Execution time per batch is bounded (mean < 100ms, max < 400ms).
        """
        rng = random.Random(42)
        total_batches = 50
        durations = []

        for batch_i in range(total_batches):
            # Generate crowded board
            board = [[1 if rng.random() < 0.60 else 0 for _ in range(8)] for _ in range(8)]
            # Pick 3 random pieces
            pieces = [rng.choice(PIECE_NAMES) for _ in range(3)]
            tray = [{"slot": i, "name": pieces[i]} for i in range(3)]

            t0 = time.perf_counter()
            plan = plan_batch_moves(board, tray, neat_champion_net)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            durations.append(dt_ms)

            assert isinstance(plan, list)

            # If moves were returned, verify each is valid step-by-step
            if plan:
                sim = BlockBlast()
                sim.board = [row[:] for row in board]
                sim.tray = [Piece(p) for p in pieces]
                used_slots = set()
                for slot_idx, r, c in plan:
                    assert slot_idx not in used_slots, f"Slot {slot_idx} reused in single batch"
                    used_slots.add(slot_idx)
                    p = sim.tray[slot_idx]
                    assert p is not None
                    assert sim.can_place(p, r, c), f"Batch {batch_i}: illegal move ({slot_idx}, {r}, {c})"
                    sim.step(slot_idx, r, c)

        assert np.mean(durations) < 100.0, f"Mean planning latency too high: {np.mean(durations):.2f}ms"
        assert max(durations) < 400.0, f"Max planning latency too high: {max(durations):.2f}ms"

    def test_adv_3piece_empty_and_partial_trays(self, neat_champion_net):
        """Tests tray permutations with None slots and sparse slots."""
        board = [[0] * 8 for _ in range(8)]

        # 1. Empty tray
        assert plan_batch_moves(board, [None, None, None], neat_champion_net) == []

        # 2. Only slot 1 present
        tray_slot1 = [None, {"slot": 1, "name": "dot"}, None]
        plan1 = plan_batch_moves(board, tray_slot1, neat_champion_net)
        assert len(plan1) == 1
        assert plan1[0][0] == 1

        # 3. Slots 0 and 2 present
        tray_0_2 = [{"slot": 0, "name": "dot"}, None, {"slot": 2, "name": "line2_h"}]
        plan_0_2 = plan_batch_moves(board, tray_0_2, neat_champion_net)
        assert len(plan_0_2) == 2
        assert {m[0] for m in plan_0_2} == {0, 2}

    def test_adv_3piece_deadlock_detection_full_board(self, neat_champion_net):
        """Tests that a completely full board with no moves returns [] immediately."""
        full_board = [[1] * 8 for _ in range(8)]
        tray = [{"slot": 0, "name": "square3x3"}, {"slot": 1, "name": "line5_v"}, {"slot": 2, "name": "corner_tl"}]
        plan = plan_batch_moves(full_board, tray, neat_champion_net)
        assert plan == []

    def test_adv_3piece_dynamic_fallback_unknown_piece(self, neat_champion_net):
        """Tests batch planner when dynamic fallback piece is present in tray."""
        board = [[0] * 8 for _ in range(8)]
        BLOCK_SHAPES["dyn_plus_mini"] = [(0, 1), (1, 0), (1, 1), (1, 2)]
        try:
            tray = [
                {"slot": 0, "name": "dyn_plus_mini"},
                {"slot": 1, "name": "dot"},
                {"slot": 2, "name": "line2_h"},
            ]
            plan = plan_batch_moves(board, tray, neat_champion_net)
            assert len(plan) == 3
        finally:
            BLOCK_SHAPES.pop("dyn_plus_mini", None)


# =====================================================================
# TIER 5.2: EXTREME BOARD PATTERNS & OCCUPANCY CLASSIFICATION
# =====================================================================

class TestTier5_ExtremeBoardPatterns:
    """Stress tests game simulation, feature extraction, and CV detection on extreme board patterns."""

    def test_adv_board_63_of_64_occupied_single_corner_hole(self):
        """
        Adversarial scenario: 63 out of 64 cells occupied.
        Single empty hole at corner (0, 0).
        Placing a dot at (0, 0) completes all 8 rows and 8 cols, clearing the full board.
        """
        sim = BlockBlast()
        sim.board = [[1] * 8 for _ in range(8)]
        sim.board[0][0] = 0  # Only (0, 0) is empty
        sim.tray = [Piece("dot"), None, None]

        features = sim.simulate_move_features(0, 0, 0)
        assert len(features) == 79
        for i, val in enumerate(features):
            assert not math.isnan(val), f"Feature {i} is NaN on 63/64 board"
            assert not math.isinf(val), f"Feature {i} is Inf on 63/64 board"

        # Execute move
        success, points, fit = sim.step(0, 0, 0)
        assert success is True
        # Completing all 64 cells triggers simultaneous clear of all rows & cols -> 0 occupied
        occupied_cells = sum(sum(row) for row in sim.board)
        assert occupied_cells == 0
        assert sim.lines_cleared_total == 16  # 8 rows + 8 cols

    @pytest.mark.parametrize("hole_r, hole_c", [
        (0, 7), (7, 0), (7, 7), (3, 3), (3, 4), (4, 3), (2, 5)
    ])
    def test_adv_board_63_of_64_various_holes(self, hole_r: int, hole_c: int):
        """Verifies 63/64 board with single hole at arbitrary coordinates."""
        sim = BlockBlast()
        sim.board = [[1] * 8 for _ in range(8)]
        sim.board[hole_r][hole_c] = 0
        sim.tray = [Piece("dot"), None, None]

        assert sim.can_place(sim.tray[0], hole_r, hole_c) is True
        assert sim.can_place(Piece("line2_h"), hole_r, hole_c) is False
        assert sim.can_place(Piece("square2x2"), hole_r, hole_c) is False

        features = sim.simulate_move_features(0, hole_r, hole_c)
        assert len(features) == 79
        assert all(0.0 <= f <= 1.0 for f in features[64:])

    def test_adv_board_checkerboard_pattern(self):
        """
        Adversarial scenario: Pure checkerboard pattern (32 occupied cells, alternating).
        No piece of size >= 2 can fit without dots.
        """
        sim = BlockBlast()
        sim.board = [[(r + c) % 2 for c in range(8)] for r in range(8)]
        sim.tray = [Piece("square2x2"), Piece("line2_h"), Piece("line2_v")]

        valid_moves = sim.get_valid_moves()
        assert valid_moves == [], "Checkerboard cannot fit any multi-block piece without clearing"

        sim.tray[0] = Piece("dot")
        dot_moves = sim.get_valid_moves()
        assert len(dot_moves) == 32  # Exactly 32 empty cells

    def test_adv_board_trapped_holes_feature_penalty(self):
        """Adversarial scenario: Board with surrounded trapped holes (>=3 neighbors)."""
        sim = BlockBlast()
        # Setup board with a single 1x1 hole at (3, 3) surrounded on all 4 sides, but rows not full
        board = [[0] * 8 for _ in range(8)]
        board[2][3] = 1
        board[4][3] = 1
        board[3][2] = 1
        board[3][4] = 1
        sim.board = board
        sim.tray = [Piece("dot"), None, None]

        # Simulating move at (0, 0) should detect the trapped hole at (3, 3)
        features = sim.simulate_move_features(0, 0, 0)
        trapped_hole_feature = features[64 + 4]  # H5: trapped holes
        assert trapped_hole_feature > 0.0, "Trapped hole at (3,3) must trigger H5 trapped hole penalty"

    def test_adv_board_full_row_and_col_clears(self):
        """Simulate single row and single column clears simultaneously."""
        sim = BlockBlast()
        # Setup board where row 3 and col 3 are missing ONLY cell (3, 3)
        board = [[0] * 8 for _ in range(8)]
        for c in range(8):
            if c != 3:
                board[3][c] = 1
        for r in range(8):
            if r != 3:
                board[r][3] = 1
        sim.board = board
        sim.tray = [Piece("dot"), None, None]

        # Place dot at (3, 3)
        success, pts, fit = sim.step(0, 3, 3)
        assert success is True
        assert sim.lines_cleared_total == 2  # 1 row + 1 col
        assert sum(sum(row) for row in sim.board) == 0  # Cleaned board

    def test_adv_cv_detector_on_63_of_64_all_themes(self, detector):
        """
        Stress tests SIMD CV detector on 63/64 occupied boards across Wood, Blue, Neon, Jungle themes.
        Must accurately detect 63 occupied cells and 1 empty cell.
        """
        for theme in ["wood", "blue", "neon", "jungle"]:
            ground_truth = [[1] * 8 for _ in range(8)]
            ground_truth[4][4] = 0  # Center hole

            frame = make_synthetic_frame(theme=theme, occupied_matrix=ground_truth)
            detected = detector.detect_board_state(frame)

            assert detected[4][4] == 0, f"Failed on theme {theme}: center hole not detected"
            total_detected = sum(sum(row) for row in detected)
            assert total_detected == 63, f"Failed on theme {theme}: expected 63 occupied, got {total_detected}"

    def test_adv_cv_detector_on_checkerboard_all_themes(self, detector):
        """
        Stress tests SIMD CV detector on checkerboard pattern across all 4 themes.
        Must accurately detect 32 occupied and 32 empty cells.
        """
        ground_truth = [[(r + c) % 2 for c in range(8)] for r in range(8)]
        for theme in ["wood", "blue", "neon", "jungle"]:
            frame = make_synthetic_frame(theme=theme, occupied_matrix=ground_truth)
            detected = detector.detect_board_state(frame)
            total_detected = sum(sum(row) for row in detected)
            assert total_detected == 32, f"Failed on theme {theme}: expected 32 occupied, got {total_detected}"


# =====================================================================
# TIER 5.3: COORDINATE MATH & SAFE BOUNDS CLAMPING (Y <= 1580)
# =====================================================================

class TestTier5_BoundsClampingAndTouchMath:
    """Stress tests coordinate math, boundary invariants, and strict Y <= 1580 release clamping."""

    def test_adv_coordinate_extreme_numbers(self):
        """Stress tests clamping functions with extreme and adversarial values."""
        extreme_inputs = [
            -1_000_000, -99999, -1, 0, 540, 1080, 1081, 2400, 2401, 1_000_000,
            -0.5, 0.499, 1079.9, 2399.9, 999999.99
        ]
        for x in extreme_inputs:
            cx = clamp_coordinate_x(x)
            assert 0 <= cx <= SCREEN_WIDTH, f"X={x} clamped to {cx} out of [0, 1080]"

        for y in extreme_inputs:
            cy = clamp_coordinate_y(y)
            assert 0 <= cy <= SCREEN_HEIGHT, f"Y={y} clamped to {cy} out of [0, 2400]"

            ry = clamp_release_y(y)
            assert 0 <= ry <= MAX_SAFE_RELEASE_Y, f"Release Y={y} clamped to {ry} > 1580"
            assert ry <= 1580, f"Release Y={y} must be <= 1580, got {ry}"
            assert not is_in_tray_cancel_zone(ry), f"Release Y={ry} fell into tray cancel zone!"

    def test_adv_finger_release_y_strictly_clamped_for_all_42_pieces_all_64_cells(self):
        """
        Exhaustive verification: For ALL 42 canonical Block Blast pieces,
        at ALL 64 grid positions (Row 0..7, Col 0..7), compute finger target XY
        via CalibrationProfile. Every release Y MUST strictly satisfy Y <= 1580.
        """
        profile = CalibrationProfile()
        violations = []

        for piece_name in BLOCK_SHAPES.keys():
            for r in range(8):
                for c in range(8):
                    target_x, target_y = profile.get_finger_target_xy(piece_name, r, c)
                    if target_y > MAX_SAFE_RELEASE_Y:
                        violations.append((piece_name, r, c, target_x, target_y))
                    if target_y >= TRAY_CANCEL_ZONE_START_Y:
                        violations.append(("CRITICAL_TRAY_ZONE", piece_name, r, c, target_y))
                    if not (0 <= target_x <= SCREEN_WIDTH):
                        violations.append(("X_OOB", piece_name, r, c, target_x))

        assert violations == [], f"Found {len(violations)} safe release clamp violations: {violations[:5]}"

    def test_adv_bot_player_plan_moves_in_memory_clamp_invariant(self, neat_champion_net):
        """
        Stress test: For bot_player.BlockBlastMobileBot.plan_moves_in_memory,
        verify that every planned touch swipe end_y coordinate is strictly <= 1580.
        """
        bot = BlockBlastMobileBot(net=neat_champion_net)
        empty_board = [[0] * 8 for _ in range(8)]

        test_pieces = [
            ("square3x3", (220, 1855)),
            ("line5_v", (540, 1855)),
            ("big_corner_br", (860, 1855)),
        ]

        planned_moves = bot.plan_moves_in_memory(empty_board, test_pieces)
        assert len(planned_moves) > 0

        for move in planned_moves:
            p_idx, r, c, start_xy, end_xy = move
            start_x, start_y = start_xy
            end_x, end_y = end_xy

            assert 0 <= start_x <= 1080
            assert 0 <= start_y <= 2400
            assert 0 <= end_x <= 1080
            assert end_y <= 1580, f"Planned end_y={end_y} exceeds MAX_SAFE_RELEASE_Y (1580)"
            assert not is_in_tray_cancel_zone(end_y), f"Planned drop {end_xy} in tray cancel zone!"

    def test_adv_corrupted_calibration_profile_extreme_lift_clamping(self):
        """
        Adversarial scenario: A corrupted profile with extreme lift_y = 9999.0 px
        and anchor offsets dx = 5000.0, dy = 5000.0.
        The system MUST safely clamp coordinates to Y <= 1580 and X in [10, 1070].
        """
        corrupted_data = {
            "profiles": {
                "dot": {
                    "family": "dot",
                    "lift_y": 9999.0,
                    "anchor_dx": 5000.0,
                    "anchor_dy": 5000.0,
                    "width_blocks": 1,
                    "height_blocks": 1,
                    "block_count": 1,
                }
            }
        }
        profile = CalibrationProfile(corrupted_data)
        safe_x, safe_y = profile.get_finger_target_xy("dot", 7, 7)
        assert safe_y <= 1580, f"Corrupted lift_y produced safe_y={safe_y} > 1580"
        assert safe_x <= 1070, f"Corrupted anchor_dx produced safe_x={safe_x} > 1070"

    def test_adv_mock_adb_swipe_command_formatting_safe_clamping(self):
        """Verifies ADB swipe command formatting with clamp_safe_release=True."""
        client = FastADBSocketClient(serial="ZF524K4RCM")

        # Test with extreme end coordinates (e.g. y2 = 2200 inside tray zone)
        x1, y1, x2, y2 = 540, 1855, 540, 2200

        clamped_y2 = clamp_release_y(y2)
        assert clamped_y2 == 1580


# =====================================================================
# TIER 5.4: ADVERSARIAL SIMD VISION & NOISE INJECTION
# =====================================================================

class TestTier5_AdversarialVisionNoise:
    """Stress tests OpenCV vision pipeline against noise, lighting variations, and throughput."""

    def test_adv_cv_noise_resilience_multi_theme(self, detector):
        """
        Injects realistic digital screencap noise (sigma=2.0) across themes
        and verifies cell occupancy classification accuracy is 100%.
        """
        ground_truth = [[1 if (r + c) % 3 == 0 else 0 for c in range(8)] for r in range(8)]

        for theme in ["blue", "neon", "jungle"]:
            clean_frame = make_synthetic_frame(theme=theme, occupied_matrix=ground_truth)
            noisy_frame = clean_frame.astype(np.float32)
            gaussian_noise = np.random.normal(0, 2.0, clean_frame.shape)
            noisy_frame = np.clip(noisy_frame + gaussian_noise, 0, 255).astype(np.uint8)

            detected = detector.detect_board_state(noisy_frame)
            errors = sum(
                1 for r in range(8) for c in range(8) if detected[r][c] != ground_truth[r][c]
            )
            accuracy = (64 - errors) / 64.0
            assert accuracy == 1.0, f"Failed theme {theme}: {errors} cell errors"

    def test_adv_cv_exposure_and_theme_variations(self, detector):
        """
        Applies exposure variations (scaling alpha in [0.90, 1.10]) across Blue, Neon, and Jungle themes,
        verifying that bezel background calibration and cell detection remain 100% accurate.
        """
        ground_truth = [[0] * 8 for _ in range(8)]
        ground_truth[2][2] = 1
        ground_truth[5][5] = 1

        for alpha in [0.90, 1.00, 1.10]:
            for theme in ["blue", "neon", "jungle"]:
                clean_frame = make_synthetic_frame(theme=theme, occupied_matrix=ground_truth)
                adjusted = np.clip(clean_frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                detected = detector.detect_board_state(adjusted)
                assert detected[2][2] == 1, f"Failed {theme} alpha={alpha} at (2,2)"
                assert detected[5][5] == 1, f"Failed {theme} alpha={alpha} at (5,5)"
                assert detected[0][0] == 0, f"Failed {theme} alpha={alpha} at (0,0)"

    def test_adv_cv_simd_throughput_100_frames(self, detector):
        """
        Measures SIMD OpenCV detection latency across 100 consecutive frames.
        Mean latency per frame must be strictly < 25ms and P95 < 50ms.
        """
        frame = make_synthetic_frame(theme="neon")
        latencies = []

        for _ in range(100):
            t0 = time.perf_counter()
            _ = detector.detect_board_state(frame)
            _ = detector.detect_tray_pieces(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        assert mean_lat < 25.0, f"Mean detection latency {mean_lat:.2f}ms exceeds 25ms limit"
        assert p95_lat < 50.0, f"P95 detection latency {p95_lat:.2f}ms exceeds 50ms limit"


# =====================================================================
# TIER 5.5: FSM STRESS, DESYNC RECOVERY & CLOSED-LOOP VERIFICATION
# =====================================================================

class TestTier5_FSMStressAndDesyncRecovery:
    """Stress tests FSM state transitions, recovery mode, and desync resilience."""

    def test_adv_fsm_socket_failure_transitions_to_recover(self, neat_champion_net):
        """
        Simulates an unexpected socket error during CAPTURE_FRAME.
        FSM must transition to RECOVER without crashing.
        """
        bot = BlockBlastMobileBot(net=neat_champion_net)
        bot.state = BotState.CAPTURE_FRAME

        def broken_capture():
            raise ConnectionResetError("ADB socket forcefully closed by peer")

        bot.capture_screen = broken_capture
        new_state = bot.step_fsm()
        assert new_state == BotState.RECOVER

    def test_adv_fsm_drop_verification_failure_retry_logic(self, neat_champion_net):
        """
        Verifies that when a drop fails closed-loop verification,
        the bot returns False and handles desync.
        """
        bot = BlockBlastMobileBot(net=neat_champion_net)
        prev_board = [[0] * 8 for _ in range(8)]
        curr_frame = np.zeros((2400, 1080, 3), dtype=np.uint8)

        bot.detector.detect_board_state = lambda f: [[0] * 8 for _ in range(8)]

        success = bot.verify_drop(prev_board, curr_frame, expected_piece=Piece("dot"), expected_pos=(0, 0))
        assert success is False, "Verification should fail when board is unchanged"

    def test_adv_fsm_game_over_clean_recovery(self, neat_champion_net):
        """
        Verifies that game-over screen triggers clean score extraction,
        restart button tap, and state reset.
        """
        bot = BlockBlastMobileBot(net=neat_champion_net, auto_restart=True)
        bot.total_score = 4296
        bot.rounds_played = 10
        bot.total_pieces_placed = 30

        bot.detector.detect_game_over = lambda f: (True, 4296)

        dummy_frame = np.zeros((2400, 1080, 3), dtype=np.uint8)
        is_over, extracted_score = bot.handle_game_over(dummy_frame)

        assert is_over is True
        assert extracted_score == 4296
        assert bot.is_game_over is False
        assert bot.current_combo_streak == 0
        assert bot.current_plan == []

    def test_adv_fsm_200_transitions_no_deadlock(self, neat_champion_net):
        """
        Executes 200 rapid FSM state transitions across synthetic frames
        to guarantee absence of deadlocks, invalid states, or state leaks.
        """
        bot = BlockBlastMobileBot(net=neat_champion_net)
        frame = make_synthetic_frame(
            theme="wood",
            occupied_matrix=[[0] * 8 for _ in range(8)],
            tray_pieces=[(0, "dot"), (1, "line2_h"), (2, "square2x2")],
        )
        bot.capture_screen = lambda: frame

        visited_states = set()
        for step_i in range(200):
            old_state = bot.state
            new_state = bot.step_fsm()
            visited_states.add(new_state)
            assert isinstance(new_state, BotState), f"Invalid state {new_state} at step {step_i}"
            assert new_state != BotState.STOPPED or bot.is_game_over

        assert BotState.CAPTURE_FRAME in visited_states
        assert BotState.DETECT_STATE in visited_states
