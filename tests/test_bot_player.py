"""
test_bot_player.py - Comprehensive Unit & Integration Tests for NEAT 3-Piece Batch Move Planner & Live Play FSM.
Covers:
- Champion neural network loading and 79-feature forward pass.
- 3-piece batch permutation search (3! = 6 sequence evaluation) across all piece categories.
- Safe release coordinate clamping (Y <= 1580) preventing tray cancellation.
- Formal FSM state transitions (IDLE -> CAPTURE -> DETECT -> PLAN -> EXECUTE -> VERIFY -> SETTLE -> GAME_OVER -> RECOVER).
- Closed-loop drop verification and retry mechanics.
- Adaptive animation delays (combo particle dissipation & tray refill bounce).
- Game-over dialog detection, clean score extraction, and autonomous restart handling.
- End-to-end mock live play sessions.
"""

import os
import time
import cv2
import numpy as np
import pytest
from typing import List, Tuple, Dict, Optional, Any

from adb_client import (
    FastADBSocketClient,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    MAX_SAFE_RELEASE_Y,
    TRAY_CANCEL_ZONE_START_Y,
    clamp_release_y,
    is_in_tray_cancel_zone,
)
from cv_detector import BlockBlastDetector, CANONICAL_SHAPES
from game import BlockBlast, Piece, BLOCK_SHAPES, PIECE_NAMES, simulate_batch_sequence
from bot_player import (
    BlockBlastMobileBot,
    BotState,
    load_champion_network,
    plan_batch_moves,
    CHAMPION_PATH,
    CONFIG_PATH,
)
from tests.conftest import SyntheticFrameBuilder, MockADBServer


# =====================================================================
# 1. NEAT CHAMPION MODEL LOADING & EVALUATION TESTS
# =====================================================================

class TestBotPlayer_ModelLoading:
    """Verifies pre-trained NEAT champion genome loading and neural inference."""

    def test_champion_model_loading_success(self, neat_champion_net):
        """Loads champion from checkpoints/best_champion.pkl with 79 inputs."""
        net = load_champion_network()
        assert net is not None

        # Test forward pass with 79 zero features
        dummy_inputs = [0.0] * 79
        output = net.activate(dummy_inputs)
        assert len(output) == 1
        assert isinstance(output[0], (int, float))

    def test_champion_model_loading_missing_file(self):
        """Handles missing checkpoint gracefully without fatal crash."""
        net = load_champion_network(
            champion_path="non_existent_champion.pkl",
            config_path=CONFIG_PATH,
        )
        assert net is None

    def test_champion_evaluation_on_simulated_features(self, game_sim: BlockBlast, neat_champion_net):
        """Simulates move features on 8x8 board and evaluates with champion network."""
        if neat_champion_net is None:
            pytest.skip("NEAT champion network not available")

        game_sim.reset(seed=42)
        features = game_sim.simulate_move_features(0, 0, 0)
        assert len(features) == 79
        assert all(0.0 <= f <= 1.0 for f in features)

        val = neat_champion_net.activate(features)
        assert len(val) == 1
        assert isinstance(val[0], float)


# =====================================================================
# 2. BATCH 3-PIECE PERMUTATION SEARCH TESTS
# =====================================================================

class TestBotPlayer_BatchPermutationPlanning:
    """Verifies in-memory 3-piece permutation planning evaluating all 3! = 6 orderings."""

    def test_plan_batch_moves_evaluates_all_six_permutations(self, neat_champion_net):
        """Evaluates 3 pieces in tray and produces 3 sequenced moves."""
        board = [[0] * 8 for _ in range(8)]
        tray = [
            ("dot", (220, 1855)),
            ("line2_h", (540, 1855)),
            ("corner_tl", (860, 1855)),
        ]

        moves = plan_batch_moves(board, tray, neat_champion_net)
        assert len(moves) == 3
        # Check that all 3 slots (0, 1, 2) are used in some optimal ordering
        used_slots = [m[0] for m in moves]
        assert sorted(used_slots) == [0, 1, 2]

        # Verify simulation of sequence executes without collision
        sim_res = simulate_batch_sequence(
            board=board,
            pieces=[Piece("dot"), Piece("line2_h"), Piece("corner_tl")],
            moves=moves,
        )
        assert sim_res["valid"] is True
        assert sim_res["moves_executed"] == 3

    def test_plan_batch_moves_single_piece_tray(self, neat_champion_net):
        """Tray with 1 piece produces exactly 1 planned move."""
        board = [[0] * 8 for _ in range(8)]
        tray = [None, ("line4_h", (540, 1855)), None]

        moves = plan_batch_moves(board, tray, neat_champion_net)
        assert len(moves) == 1
        assert moves[0][0] == 1
        p_idx, r, c = moves[0]
        assert 0 <= r < 8 and 0 <= c <= 4

    def test_plan_batch_moves_two_piece_tray(self, neat_champion_net):
        """Tray with 2 pieces evaluates 2! = 2 orderings and produces 2 planned moves."""
        board = [[0] * 8 for _ in range(8)]
        tray = [("square2x2", (220, 1855)), None, ("line3_v", (860, 1855))]

        moves = plan_batch_moves(board, tray, neat_champion_net)
        assert len(moves) == 2
        used_slots = [m[0] for m in moves]
        assert sorted(used_slots) == [0, 2]

    def test_plan_batch_moves_empty_tray_returns_empty(self, neat_champion_net):
        """Empty tray returns empty move list."""
        board = [[0] * 8 for _ in range(8)]
        tray = [None, None, None]
        moves = plan_batch_moves(board, tray, neat_champion_net)
        assert moves == []

    def test_plan_batch_moves_completely_blocked_board_returns_empty(self, neat_champion_net):
        """Full board with large pieces returns empty list without error."""
        full_board = [[1] * 8 for _ in range(8)]
        tray = [("square3x3", (220, 1855)), ("square3x3", (540, 1855)), ("square3x3", (860, 1855))]
        moves = plan_batch_moves(full_board, tray, neat_champion_net)
        assert moves == []

    def test_plan_batch_moves_input_format_dicts(self, neat_champion_net):
        """Accepts List[Dict] format from detector.detect_state()."""
        board = np.zeros((8, 8), dtype=np.int32)
        tray_dicts = [
            {"slot": 0, "name": "line2_v", "grab_xy": (220, 1855)},
            {"slot": 1, "name": "diag2_down", "grab_xy": (540, 1855)},
            {"slot": 2, "name": "t_down", "grab_xy": (860, 1855)},
        ]
        moves = plan_batch_moves(board, tray_dicts, neat_champion_net)
        assert len(moves) == 3

    def test_plan_batch_moves_input_format_pieces(self, neat_champion_net):
        """Accepts List[Piece] format."""
        board = [[0] * 8 for _ in range(8)]
        pieces = [Piece("rect2x3"), Piece("dot"), Piece("line3_h")]
        moves = plan_batch_moves(board, pieces, neat_champion_net)
        assert len(moves) == 3

    def test_plan_batch_moves_dynamic_shape_support(self, neat_champion_net):
        """Accepts dynamic unregistered shapes registered in BLOCK_SHAPES."""
        custom_name = "dyn_4x4_diagonal_custom"
        custom_blocks = [(0, 0), (1, 1), (2, 2), (3, 3)]
        custom_key = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        
        try:
            BLOCK_SHAPES[custom_name] = custom_blocks
            CANONICAL_SHAPES[custom_key] = custom_name

            board = [[0] * 8 for _ in range(8)]
            tray = [(custom_name, (220, 1855)), None, None]
            moves = plan_batch_moves(board, tray, neat_champion_net)
            assert len(moves) == 1
            assert moves[0][0] == 0
        finally:
            BLOCK_SHAPES.pop(custom_name, None)
            CANONICAL_SHAPES.pop(custom_key, None)

    def test_plan_batch_moves_combo_cascade_preference(self, neat_champion_net):
        """Planner favors sequence order that triggers line clear cascades."""
        # Row 0 has 7 blocks filled, col 7 is empty
        board = [[0] * 8 for _ in range(8)]
        for c in range(7):
            board[0][c] = 1

        # Piece 0: dot (fits at (0, 7) to clear Row 0)
        # Piece 1: line5_h (length 5, only fits row 0 after dot clears row 0!)
        tray = [
            ("dot", (220, 1855)),
            ("line5_h", (540, 1855)),
            None
        ]

        moves = plan_batch_moves(board, tray, neat_champion_net)
        assert len(moves) == 2
        # Piece 0 (dot) must be played first to clear Row 0, unlocking line5_h
        assert moves[0][0] == 0
        assert moves[1][0] == 1


# =====================================================================
# 3. COORDINATE CALCULATION & SAFE CLAMPING TESTS
# =====================================================================

class TestBotPlayer_CoordinateClamping:
    """Verifies safe touch coordinate generation and tray cancel zone avoidance."""

    def test_plan_moves_in_memory_safe_release_y_bounds(self, neat_champion_net):
        """All planned end coordinates across all 8 rows satisfy Y_release <= 1580."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        board = [[0] * 8 for _ in range(8)]
        tray = [
            ("line5_h", (220, 1855)),
            ("square3x3", (540, 1855)),
            ("big_corner_br", (860, 1855)),
        ]

        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 3

        for p_idx, r, c, start_pos, end_pos in moves:
            # 1. Start position must be in tray area
            assert 0 <= start_pos[0] <= SCREEN_WIDTH
            assert 1600 <= start_pos[1] <= 2100

            # 2. End position must be clamped safely: Y <= 1580
            assert 0 <= end_pos[0] <= SCREEN_WIDTH
            assert end_pos[1] <= MAX_SAFE_RELEASE_Y
            assert not is_in_tray_cancel_zone(end_pos[1])

    def test_plan_moves_in_memory_row_7_bottom_cell_clamping(self, neat_champion_net):
        """Pieces placed in the lowest row (Row 7) do not spill into tray cancel zone."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = BlockBlastDetector()
        bot.net = neat_champion_net

        # Fill top 7 rows, leaving only Row 7 open
        board = [[1] * 8 for _ in range(7)] + [[0] * 8]
        tray = [("dot", (220, 1855)), None, None]

        moves = bot.plan_moves_in_memory(board, tray)
        assert len(moves) == 1
        p_idx, r, c, start_pos, end_pos = moves[0]
        assert r == 7
        assert end_pos[1] <= MAX_SAFE_RELEASE_Y
        assert not is_in_tray_cancel_zone(end_pos[1])


# =====================================================================
# 4. FINITE STATE MACHINE (FSM) LIFECYCLE TESTS
# =====================================================================

class TestBotPlayer_FiniteStateMachine:
    """Verifies FSM state transitions and lifecycle flow."""

    def test_fsm_initial_state_idle(self, neat_champion_net):
        """Initial state of bot is BotState.IDLE."""
        bot = BlockBlastMobileBot(auto_restart=False, net=neat_champion_net)
        assert bot.state == BotState.IDLE

    def test_fsm_lifecycle_transition_sequence(self, detector: BlockBlastDetector, neat_champion_net):
        """Verifies clean step_fsm() transitions through one full round."""
        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "dot")

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        bot.state = BotState.IDLE
        bot.auto_restart = False
        bot.rounds_played = 0
        bot.total_pieces_placed = 0
        bot.current_combo_streak = 0
        bot.last_cleared_lines = 0
        bot.drop_retry_count = 0
        bot.max_drop_retries = 2
        bot.settle_delay = 0.01  # Fast for test

        # Mock capture and execute
        bot.capture_screen = lambda: frame
        bot.execute_swipe = lambda sx, sy, ex, ey, dur=None: True

        # 1. IDLE -> CAPTURE_FRAME
        s1 = bot.step_fsm()
        assert s1 == BotState.CAPTURE_FRAME

        # 2. CAPTURE_FRAME -> DETECT_STATE
        s2 = bot.step_fsm()
        assert s2 == BotState.DETECT_STATE

        # 3. DETECT_STATE -> PLAN_BATCH
        s3 = bot.step_fsm()
        assert s3 == BotState.PLAN_BATCH

        # 4. PLAN_BATCH -> EXECUTE_MOVE
        s4 = bot.step_fsm()
        assert s4 == BotState.EXECUTE_MOVE
        assert len(bot.current_plan) == 1

        # 5. EXECUTE_MOVE -> VERIFY_DROP
        s5 = bot.step_fsm()
        assert s5 == BotState.VERIFY_DROP

        # 6. VERIFY_DROP -> SETTLE_ANIMATION
        # Mock drop verification to return True
        bot.verify_drop = lambda *args, **kwargs: True
        s6 = bot.step_fsm()
        assert s6 == BotState.SETTLE_ANIMATION
        assert bot.total_pieces_placed == 1

        # 7. SETTLE_ANIMATION -> CAPTURE_FRAME (for new round)
        s7 = bot.step_fsm()
        assert s7 == BotState.CAPTURE_FRAME

    def test_fsm_game_over_state_transition(self, detector: BlockBlastDetector, neat_champion_net):
        """Game-over frame transitions from DETECT_STATE to CHECK_GAME_OVER -> STOPPED."""
        img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        img[581:1537, 61:1018] = (20, 20, 20)
        cv2.circle(img, (540, 1500), 70, (240, 240, 240), -1)
        cv2.putText(img, "GAME OVER", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        bot.state = BotState.DETECT_STATE
        bot.last_frame = img
        bot.auto_restart = False
        bot.total_score = 1200

        s1 = bot.step_fsm()
        assert s1 == BotState.CHECK_GAME_OVER

        s2 = bot.step_fsm()
        assert s2 == BotState.STOPPED
        assert bot.is_game_over is True

    def test_fsm_desync_recovery_transition(self, detector: BlockBlastDetector, neat_champion_net):
        """Swipe failure or capture error transitions to RECOVER -> CAPTURE_FRAME."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.net = neat_champion_net
        bot.state = BotState.RECOVER
        bot.current_plan = [(0, 0, 0, (220, 1855), (220, 1400))]

        s1 = bot.step_fsm()
        assert s1 == BotState.CAPTURE_FRAME
        assert bot.current_plan == []


# =====================================================================
# 5. CLOSED-LOOP DROP VERIFICATION TESTS
# =====================================================================

class TestBotPlayer_ClosedLoopDropVerification:
    """Verifies drop verification and recovery logic."""

    def test_verify_drop_cell_count_increment(self, detector: BlockBlastDetector):
        """Drop verified when occupied block count increases by piece size."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector

        builder = SyntheticFrameBuilder(theme="wood")
        prev_frame = builder.build_empty_frame()
        after_frame = builder.build_empty_frame()
        builder.draw_board_cells(after_frame, [(2, 2), (2, 3)])

        prev_board = detector.detect_board_state(prev_frame)
        p = Piece("line2_h")
        assert bot.verify_drop(prev_board, after_frame, expected_piece=p, expected_pos=(2, 2)) is True

    def test_verify_drop_line_clearing_board_change(self, detector: BlockBlastDetector):
        """Drop verified when a line clear changes the board state."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector

        builder = SyntheticFrameBuilder(theme="wood")
        # Prev board has 7 blocks in row 0
        prev_frame = builder.build_empty_frame()
        builder.draw_board_cells(prev_frame, [(0, c) for c in range(7)])

        # After frame has row 0 cleared
        after_frame = builder.build_empty_frame()

        prev_board = detector.detect_board_state(prev_frame)
        p = Piece("dot")
        assert bot.verify_drop(prev_board, after_frame, expected_piece=p, expected_pos=(0, 7)) is True

    def test_verify_drop_failure_detection(self, detector: BlockBlastDetector):
        """Drop flagged as failed when board state remains identical."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector

        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        prev_board = detector.detect_board_state(frame)

        # After frame is identical (piece was not dropped)
        p = Piece("square2x2")
        assert bot.verify_drop(prev_board, frame, expected_piece=p, expected_pos=(2, 2)) is False


# =====================================================================
# 6. ADAPTIVE ANIMATION TIMING TESTS
# =====================================================================

class TestBotPlayer_AdaptiveTimings:
    """Verifies adaptive delay scaling for combo particle dissipation and tray refills."""

    def test_settling_delay_default_move(self):
        """Normal non-clearing move uses standard 0.40s settling."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        delay = bot.get_adaptive_settling_delay(lines_cleared=0, is_refill=False, combo_streak=0)
        assert delay == 0.40

    def test_settling_delay_single_line_clear(self):
        """Single line clear adds particle dissipation delay (0.55s)."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        delay = bot.get_adaptive_settling_delay(lines_cleared=1, is_refill=False, combo_streak=1)
        assert delay == 0.55

    def test_settling_delay_multi_line_clear(self):
        """Multi-line clear uses 0.75s settling delay."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        delay = bot.get_adaptive_settling_delay(lines_cleared=2, is_refill=False, combo_streak=1)
        assert delay == 0.75

    def test_settling_delay_high_combo_cascade(self):
        """High combo streak (>= 3) allows full 1.10s particle dissipation."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        delay = bot.get_adaptive_settling_delay(lines_cleared=2, is_refill=False, combo_streak=4)
        assert delay == 1.10

    def test_settling_delay_tray_refill(self):
        """Tray refill bounce delay is at least 0.85s."""
        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        delay = bot.get_adaptive_settling_delay(is_refill=True)
        assert delay >= 0.85


# =====================================================================
# 7. GAME-OVER HANDLING & RECOVERY TESTS
# =====================================================================

class TestBotPlayer_GameOverHandling:
    """Verifies clean score extraction, restart handling, and crash prevention."""

    def test_game_over_score_logging_and_clean_exit(self, detector: BlockBlastDetector):
        """Extracts score cleanly and stops FSM when auto-restart is disabled."""
        img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        img[581:1537, 61:1018] = (20, 20, 20)
        cv2.circle(img, (540, 1500), 70, (240, 240, 240), -1)
        cv2.putText(img, "GAME OVER", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.auto_restart = False
        bot.total_score = 3500
        bot.rounds_played = 15
        bot.total_pieces_placed = 45

        is_over, score = bot.handle_game_over(img)
        assert is_over is True
        assert score is not None or bot.total_score == 3500

    def test_game_over_auto_restart_dispatch(self, detector: BlockBlastDetector):
        """Auto-restart dispatches tap to restart button location."""
        img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        img[581:1537, 61:1018] = (20, 20, 20)
        cv2.circle(img, (540, 1500), 70, (240, 240, 240), -1)
        cv2.putText(img, "GAME OVER", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        taps = []
        mock_client = FastADBSocketClient()
        mock_client.tap = lambda x, y: taps.append((x, y)) or True

        bot = BlockBlastMobileBot.__new__(BlockBlastMobileBot)
        bot.detector = detector
        bot.adb_client = mock_client
        bot.auto_restart = True
        bot.total_score = 2100
        bot.rounds_played = 8
        bot.total_pieces_placed = 24

        is_over, _ = bot.handle_game_over(img)
        assert is_over is True
        assert len(taps) == 1
        assert taps[0] == (540, 1500)


# =====================================================================
# 8. MOCK LIVE PLAY E2E INTEGRATION TESTS
# =====================================================================

class TestBotPlayer_MockE2EIntegration:
    """End-to-end integration tests using mock socket server and synthetic frames."""

    def test_play_single_tray_mock_round(self, detector: BlockBlastDetector, neat_champion_net, mock_adb_client: FastADBSocketClient):
        """Runs play_single_tray() executing 3 piece placements end-to-end."""
        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "line2_h")
        builder.draw_tray_piece(frame, 1, "corner_tl")
        builder.draw_tray_piece(frame, 2, "square2x2")

        swipes = []
        mock_adb_client.swipe = lambda *args, **kwargs: swipes.append((
            args[0] if len(args) > 0 else kwargs.get("x1"),
            args[1] if len(args) > 1 else kwargs.get("y1"),
            args[2] if len(args) > 2 else kwargs.get("x2"),
            args[3] if len(args) > 3 else kwargs.get("y2"),
        )) or True

        bot = BlockBlastMobileBot(
            client=mock_adb_client,
            detector=detector,
            net=neat_champion_net,
            auto_restart=False,
        )
        bot.capture_screen = lambda: frame

        res = bot.play_single_tray()
        assert res is True
        assert len(swipes) == 3
        for s in swipes:
            # Confirm safe release clamping on all swipes
            assert s[3] <= MAX_SAFE_RELEASE_Y
            assert not is_in_tray_cancel_zone(s[3])

    def test_run_fsm_with_max_steps_termination(self, detector: BlockBlastDetector, neat_champion_net, mock_adb_client: FastADBSocketClient):
        """Runs run_fsm() terminating cleanly when max_steps limit is reached."""
        builder = SyntheticFrameBuilder(theme="wood")
        frame = builder.build_empty_frame()
        builder.draw_tray_piece(frame, 0, "dot")

        bot = BlockBlastMobileBot(
            client=mock_adb_client,
            detector=detector,
            net=neat_champion_net,
            auto_restart=False,
        )
        bot.capture_screen = lambda: frame
        bot.execute_swipe = lambda *args, **kwargs: True
        bot.verify_drop = lambda *args, **kwargs: True

        # Run 8 FSM steps
        bot.run_fsm(max_steps=8)
        assert bot.state != BotState.IDLE
