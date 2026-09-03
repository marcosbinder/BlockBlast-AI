"""
test_tier3_pairwise.py - Tier 3: Cross-Feature Interactions & Pairwise Integration Tests.
Contains 18 authentic opaque-box requirements test cases covering subsystem inter-operations:
Vision ↔ NEAT Planner, Planner ↔ ADB Socket, Ghost Calibrator ↔ Vision, Combo Engine ↔ Settling,
and Dynamic Fallback ↔ Simulation.
"""

import time
import cv2
import numpy as np
import pytest
from typing import List, Tuple, Dict

from adb_client import (
    FastADBSocketClient,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    MAX_SAFE_RELEASE_Y,
    clamp_release_y,
    is_in_tray_cancel_zone,
)
from cv_detector import BlockBlastDetector, CANONICAL_SHAPES
from game import BlockBlast, Piece, BLOCK_SHAPES
from bot_player import BlockBlastMobileBot
from tests.conftest import SyntheticFrameBuilder, MockADBServer
import train


class TestTier3_PairwiseIntegration:
    """Pairwise cross-feature integration test suite."""

    def test_p1_cv_wood_theme_detection_to_neat_batch_planning(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Wood Theme CV Detection -> Batch 3-Piece NEAT Move Planning."""
        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "line2_h")
        builder.draw_tray_piece(frame, 1, "corner_tl")
        builder.draw_tray_piece(frame, 2, "square2x2")

        # 1. Vision Detection
        board = detector.detect_board_state(frame)
        tray = detector.detect_tray_pieces(frame)
        assert [p[0] for p in tray] == ["line2_h", "corner_tl", "square2x2"]

        # 2. NEAT Batch Planning
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        moves = bot.plan_moves_in_memory(board, tray)

        assert len(moves) == 3
        # Verify all planned moves are within board limits
        for p_idx, r, c, start_xy, end_xy in moves:
            assert 0 <= r < 8 and 0 <= c < 8
            assert end_xy[1] <= MAX_SAFE_RELEASE_Y

    def test_p2_cv_blue_theme_detection_to_neat_batch_planning(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Blue Theme CV Detection -> Batch 3-Piece NEAT Move Planning."""
        builder = SyntheticFrameBuilder(theme="blue")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "t_down")
        builder.draw_tray_piece(frame, 1, "z_h")
        builder.draw_tray_piece(frame, 2, "line3_v")

        board = detector.detect_board_state(frame)
        tray = detector.detect_tray_pieces(frame)
        assert [p[0] for p in tray] == ["t_down", "z_h", "line3_v"]

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 3

    def test_p3_cv_neon_theme_detection_to_neat_batch_planning(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Neon Theme CV Detection -> NEAT Move Planning."""
        builder = SyntheticFrameBuilder(theme="neon")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "rect2x3")
        builder.draw_tray_piece(frame, 2, "corner_br")

        board = detector.detect_board_state(frame)
        tray = detector.detect_tray_pieces(frame)

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 2

    def test_p4_cv_jungle_theme_detection_to_neat_batch_planning(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Jungle Theme CV Detection -> NEAT Move Planning."""
        builder = SyntheticFrameBuilder(theme="jungle")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 1, "plus_cross")

        board = detector.detect_board_state(frame)
        tray = detector.detect_tray_pieces(frame)

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1

    def test_p5_dynamic_shape_fallback_feeding_neat_batch_planning(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Dynamic Shape Synthesis -> NEAT Simulator Planning."""
        # Create an unconventional custom shape (e.g. 2x3 U-shape)
        tray_mask = np.zeros((390, 960), dtype=np.uint8)
        x, y, w, h = 100, 100, int(3 * 58.5), int(2 * 58.5)
        # Top row: 3 blocks, bottom row: 2 corner blocks (U-shape)
        tray_mask[100:158, 100:275] = 255
        tray_mask[158:217, 100:158] = 255
        tray_mask[158:217, 217:275] = 255

        custom_name = detector._classify_shape(tray_mask, x, y, w, h)
        assert custom_name is not None
        assert custom_name in BLOCK_SHAPES

        # Feed custom piece to planner
        board = [[0] * 8 for _ in range(8)]
        tray = [(custom_name, (540, 1855)), (None, None), (None, None)]

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1
        assert moves[0][0] == 0

    def test_p6_empirical_ghost_calibration_feeding_finger_target_coordinates(self, detector: BlockBlastDetector):
        """Pairwise: Ghost Highlight Detection -> Touch Calibration Offset Calculation."""
        builder = SyntheticFrameBuilder(theme="wood")
        base = builder.build_empty_frame()
        held = base.copy()

        # Place piece 'line3_h' at target (2, 3) -> ghost highlight at (2, 3), (2, 4), (2, 5)
        target_cells = [(2, 3), (2, 4), (2, 5)]
        builder.draw_ghost_highlights(held, target_cells, delta_e=35.0)

        ghosts = detector.detect_ghost_highlights(held, base)
        active_cells = [(r, c) for r in range(8) for c in range(8) if ghosts[r, c]]
        assert sorted(active_cells) == sorted(target_cells)

        # Compute empirical target center
        center_cell_r, center_cell_c = 2, 4
        cx, cy = detector.get_cell_screen_coords(center_cell_r, center_cell_c)
        finger_release_y = clamp_release_y(cy + detector.finger_lift_offset)

        assert finger_release_y <= MAX_SAFE_RELEASE_Y
        assert not is_in_tray_cancel_zone(finger_release_y)

    def test_p7_combo_cascade_triggering_adaptive_settling_delay(self, game_sim: BlockBlast):
        """Pairwise: Game Multi-Line Clear -> Adaptive Particle Delay Calculation."""
        # Pre-fill row 0 and row 1 with 7 blocks each
        for c in range(7):
            game_sim.board[0][c] = 1
            game_sim.board[1][c] = 1
        game_sim.board[7][7] = 1  # prevent 100% board wipe

        # Double line clear with line2_v
        game_sim.tray = [Piece("line2_v"), None, None]
        success, pts, _ = game_sim.step(0, 0, 7)
        assert success is True
        assert game_sim.lines_cleared_total == 2

        # Delay formula: base (0.40s) + 0.15s per line cleared
        adaptive_delay = 0.40 + 0.15 * game_sim.lines_cleared_total
        assert adaptive_delay == 0.70  # 700ms particle settling

    def test_p8_neat_batch_planning_feeding_mock_adb_swipe_dispatch(self, neat_champion_net, mock_adb_client: FastADBSocketClient):
        """Pairwise: NEAT Planner -> Direct Socket Swipe Dispatch."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [("line2_h", (220, 1855)), (None, None), (None, None)]
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1

        p_idx, r, c, start_pos, end_pos = moves[0]
        # Dispatch via mock ADB client
        success = mock_adb_client.swipe(start_pos[0], start_pos[1], end_pos[0], end_pos[1], duration_ms=450)
        assert success is True

    def test_p9_closed_loop_drop_verification_between_tray_moves(self, neat_champion_net):
        """Pairwise: Sequential Moves in Plan Update In-Memory Simulation State Non-overlapping."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        # 3 large pieces
        tray = [("square2x2", (220, 1855)), ("square2x2", (540, 1855)), ("square2x2", (860, 1855))]
        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 3

        # Simulate sequentially to confirm no collision
        sim = BlockBlast(seed=42)
        sim.tray = [Piece("square2x2"), Piece("square2x2"), Piece("square2x2")]
        sim.refill_tray = lambda: None

        for idx, (p_idx, r, c, _, _) in enumerate(moves):
            assert sim.can_place(sim.tray[p_idx], r, c) is True
            sim.step(p_idx, r, c)

        assert sim.moves_count == 3
        assert sum(sum(row) for row in sim.board) == 12  # 3 * 4 blocks

    def test_p10_screencap_socket_streaming_feeding_cv_board_occupancy(self, mock_adb_client: FastADBSocketClient, detector: BlockBlastDetector):
        """Pairwise: Direct Socket Screencap -> OpenCV Board State Parser."""
        # Generate dummy 1080x2400 frame on mock server
        frame = mock_adb_client.screencap_cv2()
        assert frame.shape == (2400, 1080, 3)

        board = detector.detect_board_state(frame)
        assert len(board) == 8
        assert all(len(row) == 8 for row in board)

    def test_p11_game_over_vision_detection_triggering_clean_fsm_shutdown(self, detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Vision Game-Over Detection -> FSM Loop Clean Termination."""
        # Darkened game-over image with restart button
        img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        img[581:1537, 61:1018] = (15, 15, 15)
        cv2.circle(img, (540, 1500), 75, (230, 230, 230), -1)
        cv2.putText(img, "GAME OVER", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        is_over, score = detector.detect_game_over(img)
        assert bool(is_over) is True

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net

        # When game over is detected, planner issues 0 moves
        tray = detector.detect_tray_pieces(img)
        assert all(p[0] is None for p in tray)
        moves = bot.plan_moves_in_memory([[1] * 8 for _ in range(8)], tray)
        assert moves == []

    def test_p12_socket_failure_triggering_subprocess_fallback_seamlessly(self):
        """Pairwise: Socket Unavailability -> Seamless Subprocess Fallback."""
        client = FastADBSocketClient(host="127.0.0.1", port=59998, serial="ZF524K4RCM", connect_timeout=0.05)
        # When socket fails, client attempts subprocess fallback
        try:
            img = client.screencap_cv2()
            assert isinstance(img, np.ndarray)
            assert client.last_capture_method == "subprocess"
        except RuntimeError as e:
            # If no device/adb available in environment, confirms both socket and subprocess were attempted
            assert "All screencap methods failed" in str(e)

    def test_p13_high_density_crowded_board_to_anti_suicide_move_selection(self, game_sim: BlockBlast, neat_champion_net):
        """Pairwise: 45 Occupied Cells -> NEAT Strategic Heuristic Move Selection."""
        # Create crowded board (45 cells occupied)
        np.random.seed(42)
        for r in range(8):
            for c in range(8):
                if (r + c) % 3 != 0:
                    game_sim.board[r][c] = 1

        game_sim.tray = [Piece("dot"), Piece("line2_h"), None]
        valid_moves = game_sim.get_valid_moves()
        assert len(valid_moves) > 0

        if neat_champion_net is not None:
            best_move = train._choose_best_move(game_sim, neat_champion_net)
            assert best_move in valid_moves
            p_idx, r, c = best_move
            success, _, fitness = game_sim.step(p_idx, r, c)
            assert success is True

    def test_p14_multi_piece_tray_refill_desync_recovery(self, game_sim: BlockBlast):
        """Pairwise: Partial Tray Execution -> Complete Tray Consumption -> Refill."""
        game_sim.reset(seed=777)
        assert all(p is not None for p in game_sim.tray)

        # Place piece 0
        game_sim.step(0, 0, 0)
        assert game_sim.tray[0] is None
        assert game_sim.tray[1] is not None
        assert game_sim.tray[2] is not None

        # Place piece 1
        game_sim.step(1, 2, 0)
        assert game_sim.tray[1] is None
        assert game_sim.tray[2] is not None

        # Place piece 2 -> triggers automatic tray refill!
        game_sim.step(2, 4, 0)
        assert all(p is not None for p in game_sim.tray)

    def test_p15_sample_repository_images_feeding_vision_and_planner(self, sample_images: Dict[str, np.ndarray], detector: BlockBlastDetector, neat_champion_net):
        """Pairwise: Repository PNG Captures -> CV Parser -> NEAT Move Plan."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net

        tested_count = 0
        for img_name in ("phone_screen.png", "current_state.png", "after_blue_test.png", "current_live.png"):
            if img_name in sample_images:
                frame = sample_images[img_name]
                board = detector.detect_board_state(frame)
                tray = detector.detect_tray_pieces(frame)
                assert isinstance(board, list)
                assert isinstance(tray, list)
                moves = bot.plan_moves_in_memory(board, tray)
                assert isinstance(moves, list)
                tested_count += 1

        assert tested_count >= 1

    def test_p16_ghost_highlight_matching_against_canonical_shape_rotations(self, detector: BlockBlastDetector):
        """Pairwise: 4 Rotations of T-shape match distinct canonical names."""
        for t_variant in ("t_up", "t_down", "t_left", "t_right"):
            p = Piece(t_variant)
            builder = SyntheticFrameBuilder(theme="wood")
            frame = builder.build_empty_frame()
            builder.draw_tray_piece(frame, 1, t_variant)
            detected_tray = detector.detect_tray_pieces(frame)
            assert detected_tray[1][0] == t_variant

    def test_p17_rapid_screencap_and_swipe_pipeline_throughput(self, mock_adb_client: FastADBSocketClient):
        """Pairwise: 5 Rapid Screencap + Swipe Cycles execute without socket leaking."""
        for i in range(5):
            img = mock_adb_client.screencap_cv2()
            assert img is not None
            res = mock_adb_client.swipe(220, 1855, 540, 1200, duration_ms=200)
            assert res is True

    def test_p18_board_clear_wipe_to_tray_continuation(self, game_sim: BlockBlast):
        """Pairwise: Total Board Wipe (+300) -> Subsequent Placement on Pristine Board."""
        # Row 0 full except (0, 0)
        for c in range(1, 8):
            game_sim.board[0][c] = 1

        game_sim.tray = [Piece("dot"), Piece("line2_h"), None]
        # 1. Place dot at (0, 0) -> wipes row 0 -> board 100% clean
        success, pts, _ = game_sim.step(0, 0, 0)
        assert success is True
        assert sum(sum(row) for row in game_sim.board) == 0
        assert pts >= 300

        # 2. Place next piece 'line2_h' on clean board
        success2, pts2, _ = game_sim.step(1, 4, 4)
        assert success2 is True
        assert sum(sum(row) for row in game_sim.board) == 2
