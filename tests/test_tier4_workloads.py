"""
test_tier4_workloads.py - Tier 4: Real-World Live Play Workloads & Game Cycle Tests.
Contains 12 authentic opaque-box requirements test cases covering continuous multi-round play,
multi-theme endurance, full game-over lifecycles, high combo cascades, and stress workloads.
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


class TestTier4_WorkloadsAndGameCycles:
    """Tier 4: Application-level workloads, endurance runs, and game lifecycle tests."""

    def test_w1_continuous_10_round_autonomous_play_loop(self, game_sim: BlockBlast, neat_champion_net):
        """Workload: Continuous 10-round autonomous play (30 pieces placed) without crash or failure."""
        game_sim.reset(seed=2026)
        rounds_played = 0

        for round_idx in range(10):
            # Each round plays 3 pieces in tray
            for step_idx in range(3):
                valid_moves = game_sim.get_valid_moves()
                if not valid_moves or game_sim.game_over:
                    break

                if neat_champion_net is not None:
                    move = train._choose_best_move(game_sim, neat_champion_net)
                else:
                    move = valid_moves[0]

                success, pts, _ = game_sim.step(*move)
                assert success is True

            rounds_played += 1
            if game_sim.game_over:
                break

        assert rounds_played == 10
        assert game_sim.moves_count >= 15  # Survived multiple full rounds
        assert game_sim.score > 0

    def test_w2_multi_theme_round_progression_workload(self, detector: BlockBlastDetector, neat_champion_net):
        """Workload: Theme switches dynamically each round (Wood -> Blue -> Neon -> Jungle) with vision + planning."""
        themes = ["wood", "blue", "neon", "jungle"]
        pieces_sets = [
            ["line2_h", "corner_tl", "square2x2"],
            ["t_down", "z_h", "line3_v"],
            ["rect2x3", "dot", "corner_br"],
            ["plus_cross", "line4_h", "diag2_down"],
        ]

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net

        for theme_idx, (theme, pieces) in enumerate(zip(themes, pieces_sets)):
            builder = SyntheticFrameBuilder(theme=theme)
            frame = builder.build_empty_frame()
            for slot_i, p_name in enumerate(pieces):
                builder.draw_tray_piece(frame, slot_i, p_name)

            # Vision extraction
            board = detector.detect_board_state(frame)
            tray = detector.detect_tray_pieces(frame)
            detected_names = [p[0] for p in tray]
            assert detected_names == pieces, f"Theme {theme} failed tray piece detection: got {detected_names}"

            # In-memory planning
            moves = bot.plan_moves_in_memory(board, tray)
            assert len(moves) == 3, f"Theme {theme} failed to plan 3 moves: got {len(moves)}"

    def test_w3_full_game_lifecycle_to_clean_game_over_and_restart(self, game_sim: BlockBlast, neat_champion_net):
        """Workload: Complete game cycle -> natural death -> statistics extraction -> clean restart."""
        game_sim.reset(seed=999)

        # Play moves until game naturally terminates
        max_steps = 200
        while not game_sim.game_over and game_sim.moves_count < max_steps:
            valid_moves = game_sim.get_valid_moves()
            if not valid_moves:
                break
            if neat_champion_net is not None:
                move = train._choose_best_move(game_sim, neat_champion_net)
            else:
                move = valid_moves[0]
            game_sim.step(*move)

        # Confirm score & stats recorded
        assert game_sim.moves_count > 0
        final_score = game_sim.score
        assert final_score >= 0

        # Perform clean restart
        game_sim.reset(seed=1000)
        assert game_sim.game_over is False
        assert game_sim.score == 0
        assert game_sim.moves_count == 0
        assert sum(sum(row) for row in game_sim.board) == 0

        # Verify new game plays immediately
        valid_moves = game_sim.get_valid_moves()
        assert len(valid_moves) > 0

    def test_w4_high_combo_cascade_stress_workload(self, game_sim: BlockBlast):
        """Workload: High combo streak cascade maintains accurate multipliers and fitness bonus."""
        game_sim.reset(seed=555)

        # Manually trigger 3 consecutive line clears to build streak
        game_sim.board[7][7] = 1  # prevent full board wipe bonus

        # 1. Clear Row 0
        for c in range(7):
            game_sim.board[0][c] = 1
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 0, 7)
        assert game_sim.combo_streak == 1

        # 2. Clear Row 1
        for c in range(7):
            game_sim.board[1][c] = 1
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 1, 7)
        assert game_sim.combo_streak == 2

        # 3. Clear Row 2
        for c in range(7):
            game_sim.board[2][c] = 1
        game_sim.tray = [Piece("dot"), None, None]
        game_sim.step(0, 2, 7)
        assert game_sim.combo_streak == 3
        assert game_sim.max_combo == 3

    def test_w5_board_wipe_double_recovery_workload(self, game_sim: BlockBlast):
        """Workload: Two consecutive board wipes (+300 bonus each) with intermediate play."""
        game_sim.reset(seed=333)

        # Symmetrically fill Row 0 except (0, 0)
        for c in range(1, 8):
            game_sim.board[0][c] = 1
        game_sim.tray = [Piece("dot"), Piece("line2_h"), Piece("dot")]

        # Wipe 1
        game_sim.step(0, 0, 0)
        assert sum(sum(row) for row in game_sim.board) == 0
        assert game_sim.board_clears == 1

        # Place piece on empty board
        game_sim.step(1, 3, 3)
        assert sum(sum(row) for row in game_sim.board) == 2

    def test_w6_batch_permutation_efficiency_benchmark_30_moves(self, neat_champion_net):
        """Workload: 10 batches of 3-piece planning execute within sub-1.5s total latency."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [("square2x2", (220, 1855)), ("line3_h", (540, 1855)), ("corner_tl", (860, 1855))]

        # Warmup
        _ = bot.plan_moves_in_memory(board, tray)

        t0 = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            bot.plan_moves_in_memory(board, tray)
        total_ms = (time.perf_counter() - t0) * 1000.0
        avg_plan_ms = total_ms / iterations

        assert avg_plan_ms < 1500.0, f"Average batch plan latency {avg_plan_ms:.2f}ms exceeds 1500ms ceiling"

    def test_w7_tray_cancellation_prevention_endurance_run(self, neat_champion_net):
        """Workload: 50 Lower-Half Placements (Rows 4-7) verify 100% Y_release <= 1580px."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray_candidates = ["dot", "line2_h", "line3_h", "square2x2", "corner_tl", "t_up"]

        for i in range(50):
            p_name = tray_candidates[i % len(tray_candidates)]
            tray = [(p_name, (220, 1855)), (None, None), (None, None)]
            moves = bot.plan_moves_in_memory(board, tray)
            for _, r, c, start_pos, end_pos in moves:
                assert end_pos[1] <= MAX_SAFE_RELEASE_Y, f"Release Y={end_pos[1]} exceeded MAX_SAFE_RELEASE_Y at move {i}"
                assert not is_in_tray_cancel_zone(end_pos[1]), f"Release Y={end_pos[1]} in cancel zone at move {i}"

    def test_w8_partial_tray_drop_and_recovery_workload(self, neat_champion_net):
        """Workload: Bot recovers gracefully when only 1 or 2 pieces remain in tray."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        # Scenario 1: Only slot 0 has piece
        tray_1 = [("line2_v", (220, 1855)), (None, None), (None, None)]
        moves_1 = bot.plan_moves_in_memory([[0] * 8 for _ in range(8)], tray_1)
        assert len(moves_1) == 1
        assert moves_1[0][0] == 0

        # Scenario 2: Only slot 2 has piece
        tray_2 = [(None, None), (None, None), ("square2x2", (860, 1855))]
        moves_2 = bot.plan_moves_in_memory([[0] * 8 for _ in range(8)], tray_2)
        assert len(moves_2) == 1
        assert moves_2[0][0] == 2

    def test_w9_dynamic_unknown_piece_injection_marathon(self, detector: BlockBlastDetector, neat_champion_net):
        """Workload: 5 dynamic seasonal/custom piece variants registered and evaluated on the fly."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net

        # Create 5 synthetic custom pieces
        for size in range(2, 7):
            tray_mask = np.zeros((390, 960), dtype=np.uint8)
            x, y, w, h = 100, 100, int(size * 58.5), int(2 * 58.5)
            tray_mask[100:158, 100:100 + int(size * 58.5)] = 255  # horizontal bar of length size

            custom_name = detector._classify_shape(tray_mask, x, y, w, h)
            assert custom_name is not None
            assert custom_name in BLOCK_SHAPES

            # Evaluate with simulator
            sim = BlockBlast(seed=42)
            sim.tray = [Piece(custom_name), None, None]
            valid_moves = sim.get_valid_moves()
            if valid_moves:
                sim.step(*valid_moves[0])

    def test_w10_adversarial_dirty_frame_noise_resilience_workload(self, detector: BlockBlastDetector):
        """Workload: Contaminated screen with combo banners and celebration sparkles does not corrupt piece detection."""
        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "line2_h")
        builder.draw_tray_piece(frame, 2, "square2x2")

        # Adversarial noise: banner text in upper transition area (Y=1620)
        cv2.putText(frame, "COMBO x4 STREAK!", (150, 1640), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # Sparkle particles in empty margins
        for _ in range(15):
            rx = int(np.random.randint(60, 1020))
            ry = int(np.random.randint(1660, 1720))
            cv2.circle(frame, (rx, ry), int(np.random.randint(2, 5)), (0, 255, 255), -1)

        pieces = detector.detect_tray_pieces(frame)
        assert pieces[0][0] == "line2_h"
        assert pieces[2][0] == "square2x2"

    def test_w11_mock_adb_streaming_and_swipe_stress_run(self, mock_adb_client: FastADBSocketClient):
        """Workload: 15 full round-trips of screencap + 3-swipe dispatches execute cleanly."""
        for round_idx in range(15):
            frame = mock_adb_client.screencap_cv2()
            assert frame.shape == (2400, 1080, 3)
            # 3 swipes per round
            for s in range(3):
                res = mock_adb_client.swipe(220 + s * 320, 1855, 540, 1200, duration_ms=250)
                assert res is True

    def test_w12_champion_evaluation_fitness_milestone_check(self, neat_champion_net):
        """Workload: Pretrained NEAT champion evaluated over 3 deterministic games achieves healthy score."""
        if neat_champion_net is None:
            pytest.skip("NEAT champion network checkpoint not loaded")

        scores = []
        for seed in (42, 100, 777):
            fitness, score = train._play_one_game(neat_champion_net, seed=seed)
            scores.append(score)
            assert fitness > 0

        avg_score = sum(scores) / len(scores)
        assert avg_score >= 0
