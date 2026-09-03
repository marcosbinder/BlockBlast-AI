"""
bot_player.py - Autonomous High-Speed NEAT Batch Permutation Planner & Live Play FSM.
Target Hardware: Motorola Moto G54 5G (1080x2400) via Direct ADB Socket.

Features:
- Pre-trained NEAT champion neural network (Gen 137, 79 inputs) loading and evaluation.
- Batch 3-piece permutation planning (3! = 6 sequence permutations) in-memory.
- Finite State Machine (FSM): IDLE -> CAPTURE_FRAME -> DETECT_STATE -> PLAN_BATCH -> EXECUTE_MOVE -> VERIFY_DROP -> SETTLE_ANIMATION -> CHECK_GAME_OVER -> RECOVER.
- Closed-loop drop verification with retry capability and desync recovery.
- Adaptive animation delays (combo particle dissipation scaling & tray refill bounce).
- Safe physical coordinate clamping (Y <= 1580px, avoiding tray cancel zone Y >= 1600px).
- Clean game-over dialog detection, score extraction, and autonomous restart handling.
"""

from __future__ import annotations

import enum
import itertools
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import neat
import numpy as np

from adb_client import (
    DEFAULT_ADB_HOST,
    DEFAULT_ADB_PORT,
    DEFAULT_DEVICE_SERIAL,
    MAX_SAFE_RELEASE_Y,
    FastADBSocketClient,
    clamp_coordinate_x,
    clamp_coordinate_y,
    clamp_release_y,
    find_adb_executable,
    is_in_tray_cancel_zone,
)
from cv_detector import BlockBlastDetector
from game import BLOCK_SHAPES, BlockBlast, Piece, simulate_batch_sequence

logger = logging.getLogger("bot_player")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAMPION_PATH = os.path.join(WORKSPACE_DIR, "checkpoints", "best_champion.pkl")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "config-feedforward")
ADB_PATH = r"C:\Users\Marco\Documents\scrcpy-win64-v3.3.4\adb.exe"
DEVICE_SERIAL = DEFAULT_DEVICE_SERIAL


class BotState(enum.Enum):
    """Formal Finite State Machine (FSM) States for Live Play Loop."""
    IDLE = "IDLE"
    CAPTURE_FRAME = "CAPTURE_FRAME"
    DETECT_STATE = "DETECT_STATE"
    PLAN_BATCH = "PLAN_BATCH"
    EXECUTE_MOVE = "EXECUTE_MOVE"
    VERIFY_DROP = "VERIFY_DROP"
    SETTLE_ANIMATION = "SETTLE_ANIMATION"
    CHECK_GAME_OVER = "CHECK_GAME_OVER"
    RECOVER = "RECOVER"
    STOPPED = "STOPPED"


def load_champion_network(
    champion_path: str = CHAMPION_PATH,
    config_path: str = CONFIG_PATH,
) -> Optional[neat.nn.FeedForwardNetwork]:
    """
    Loads the pre-trained NEAT champion neural network from checkpoint pickle.
    Returns FeedForwardNetwork instance or None if not found / load fails.
    """
    if not os.path.exists(champion_path):
        logger.warning(f"Champion checkpoint not found at: {champion_path}")
        return None
    if not os.path.exists(config_path):
        logger.warning(f"NEAT config file not found at: {config_path}")
        return None

    try:
        with open(champion_path, "rb") as f:
            data = pickle.load(f)

        champion_genome = data["genome"]
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        net = neat.nn.FeedForwardNetwork.create(champion_genome, config)
        gen = data.get("generation", "?")
        score = data.get("score", "?")
        fitness = data.get("fitness", "?")
        logger.info(f"Loaded NEAT Champion | Gen {gen} | Score: {score} | Fitness: {fitness}")
        return net
    except Exception as e:
        logger.error(f"Failed to load champion network ({e})")
        return None


def _fast_simulate_placement(
    board: List[List[int]],
    piece: Piece,
    r: int,
    c: int,
) -> Optional[Tuple[List[List[int]], int, int, int]]:
    """
    High-speed in-memory placement simulation.
    Returns (new_board, points, cleared_lines, contacts) or None if collision/OOB.
    """
    # 1. Bounds & collision check
    for dr, dc in piece.blocks:
        pr = r + dr
        pc = c + dc
        if pr < 0 or pr >= 8 or pc < 0 or pc >= 8 or board[pr][pc] != 0:
            return None

    # 2. Clone board and place blocks + calculate contacts
    new_board = [row[:] for row in board]
    contacts = 0
    for dr, dc in piece.blocks:
        pr = r + dr
        pc = c + dc
        new_board[pr][pc] = 1
        if pr == 0 or pr == 7: contacts += 1
        if pc == 0 or pc == 7: contacts += 1
        if pr > 0 and board[pr - 1][pc] != 0: contacts += 1
        if pr < 7 and board[pr + 1][pc] != 0: contacts += 1
        if pc > 0 and board[pr][pc - 1] != 0: contacts += 1
        if pc < 7 and board[pr][pc + 1] != 0: contacts += 1

    # 3. Fast line clearing
    rows_to_clear = [row_idx for row_idx in range(8) if all(new_board[row_idx][col_idx] != 0 for col_idx in range(8))]
    cols_to_clear = [col_idx for col_idx in range(8) if all(new_board[row_idx][col_idx] != 0 for row_idx in range(8))]
    cleared = len(rows_to_clear) + len(cols_to_clear)

    for row_idx in rows_to_clear:
        for col_idx in range(8):
            new_board[row_idx][col_idx] = 0
    for col_idx in cols_to_clear:
        for row_idx in range(8):
            new_board[row_idx][col_idx] = 0

    points = piece.size
    if cleared > 0:
        points += 10 * (cleared * (cleared + 1)) // 2
        if all(new_board[row_idx][col_idx] == 0 for row_idx in range(8) for col_idx in range(8)):
            points += 300

    return new_board, points, cleared, contacts


def plan_batch_moves(
    board_state: Union[np.ndarray, List[List[int]]],
    tray_pieces: Union[
        List[Dict[str, Any]],
        List[Optional[Piece]],
        List[Tuple[Optional[str], Optional[Tuple[int, int]]]],
    ],
    neat_net: Optional[neat.nn.FeedForwardNetwork] = None,
    beam_width: int = 8,
) -> List[Tuple[int, int, int]]:
    """
    Evaluates all sequence permutations (up to 3! = 6 paths) in-memory across
    all valid board placement positions, scoring candidate paths using the NEAT
    champion network to pick the globally optimal multi-piece combo sequence.
    """
    # 1. Normalize board to 8x8 list of lists
    if isinstance(board_state, np.ndarray):
        board = board_state.astype(int).tolist()
    else:
        board = [list(row) for row in board_state]

    # 2. Normalize tray pieces to dict mapping slot_idx -> Piece
    slot_pieces: Dict[int, Piece] = {}
    for slot_idx, item in enumerate(tray_pieces):
        if item is None:
            continue
        if isinstance(item, dict):
            slot = item.get("slot", slot_idx)
            name = item.get("name")
            if name and name in BLOCK_SHAPES:
                slot_pieces[slot] = Piece(name)
        elif isinstance(item, (tuple, list)):
            p_name = item[0]
            if p_name is not None and isinstance(p_name, str) and p_name in BLOCK_SHAPES:
                slot_pieces[slot_idx] = Piece(p_name)
        elif isinstance(item, Piece):
            slot_pieces[slot_idx] = item
        elif isinstance(item, str) and item in BLOCK_SHAPES:
            slot_pieces[slot_idx] = Piece(item)

    active_slots = list(slot_pieces.keys())
    if not active_slots:
        return []

    # If only 1 piece, evaluate all valid placements
    if len(active_slots) == 1:
        s0 = active_slots[0]
        piece0 = slot_pieces[s0]
        sim0 = BlockBlast()
        sim0.board = [row[:] for row in board]
        sim0.tray = [None, None, None]
        sim0.tray[s0] = piece0
        valid_moves = sim0.get_valid_moves_for_piece(s0)
        if not valid_moves:
            return []
        best_val = -float("inf")
        best_m = valid_moves[0]
        for m in valid_moves:
            features = sim0.simulate_move_features(m[0], m[1], m[2])
            val = neat_net.activate(features)[0] if neat_net is not None else 0.0
            sim_step = sim0.clone()
            _, pts, step_fit = sim_step.step(m[0], m[1], m[2])
            total_val = val + step_fit + pts
            if total_val > best_val:
                best_val = total_val
                best_m = m
        return [best_m]

    # Generate all sequence permutations (e.g. 2 for 2 pieces, 6 for 3 pieces)
    permutations = list(itertools.permutations(active_slots))
    candidate_sequences: List[Tuple[float, List[Tuple[int, int, int]], List[List[int]]]] = []

    for perm in permutations:
        k = len(perm)
        # Current beams: list of (cumulative_score, board_state, path)
        current_beams: List[Tuple[float, List[List[int]], List[Tuple[int, int, int]]]] = [
            (0.0, [row[:] for row in board], [])
        ]

        for depth, slot_idx in enumerate(perm):
            piece = slot_pieces[slot_idx]
            next_beams = []
            max_r = 9 - piece.height
            max_c = 9 - piece.width

            for acc_score, b_state, path in current_beams:
                candidates = []
                for r in range(max_r):
                    for c in range(max_c):
                        sim_res = _fast_simulate_placement(b_state, piece, r, c)
                        if sim_res is None:
                            continue

                        new_b, pts, cleared, contacts = sim_res
                        max_contacts = max(1, piece.size * 4)
                        contact_fit = (contacts / max_contacts) * 3.0
                        step_score = pts + cleared * 50.0 + contact_fit

                        candidates.append((step_score, new_b, path + [(slot_idx, r, c)]))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for c_val, c_board, c_path in candidates[:beam_width]:
                        next_beams.append((acc_score + c_val, c_board, c_path))
                else:
                    # Partial path dead-end
                    deadlock_penalty = -500.0 * (k - depth)
                    next_beams.append((acc_score + deadlock_penalty, b_state, path))

            if next_beams:
                next_beams.sort(key=lambda x: x[0], reverse=True)
                current_beams = next_beams[:beam_width]
            else:
                break

        if current_beams:
            top_score, final_b, top_path = current_beams[0]
            if len(top_path) == k:
                top_score += 500.0  # Full sequence completion bonus
            candidate_sequences.append((top_score, top_path, final_b))

    if not candidate_sequences:
        return []

    # Final scoring: evaluate top candidate sequences with NEAT champion network if available
    best_path: List[Tuple[int, int, int]] = []
    best_final_score = -float("inf")

    # Evaluate the top paths
    candidate_sequences.sort(key=lambda x: x[0], reverse=True)
    top_eval_pool = candidate_sequences[:6]

    sim_eval = BlockBlast()
    for base_score, path, final_board in top_eval_pool:
        if not path:
            continue
        final_score = base_score
        if neat_net is not None and len(path) == len(active_slots):
            # Extract features on final resulting board
            sim_eval.board = [row[:] for row in final_board]
            sim_eval.tray = [None, None, None]
            # Simulate features for dummy slot to get final board evaluation
            features = sim_eval.simulate_move_features(0, 0, 0)
            net_val = neat_net.activate(features)[0]
            final_score += net_val * 20.0

        if final_score > best_final_score:
            best_final_score = final_score
            best_path = path

    return best_path


class BlockBlastMobileBot:
    """
    High-Speed Autonomous NEAT Champion Live Play Engine & FSM Loop.
    """

    def __init__(
        self,
        adb_path: Optional[str] = None,
        serial: str = DEFAULT_DEVICE_SERIAL,
        host: str = DEFAULT_ADB_HOST,
        port: int = DEFAULT_ADB_PORT,
        auto_restart: bool = True,
        client: Optional[FastADBSocketClient] = None,
        detector: Optional[BlockBlastDetector] = None,
        net: Optional[neat.nn.FeedForwardNetwork] = None,
    ) -> None:
        self.serial = serial
        self.auto_restart = auto_restart

        # ADB Socket Client
        if client is not None:
            self.adb_client = client
        else:
            resolved_adb = find_adb_executable(adb_path or ADB_PATH)
            self.adb_client = FastADBSocketClient(
                serial=self.serial,
                host=host,
                port=port,
                adb_path=resolved_adb,
            )

        # Vision Detector
        self.detector = detector if detector is not None else BlockBlastDetector()

        # NEAT Champion Network
        if net is not None:
            self.net = net
        else:
            self.net = self._load_champion_network()

        # Calibration parameters (Moto G54 5G)
        self.finger_lift_offset = 205
        self.swipe_duration_ms = 450

        # FSM State & Session Telemetry
        self.state: BotState = BotState.IDLE
        self.current_plan: List[Tuple[int, int, int, Tuple[int, int], Tuple[int, int]]] = []
        self.current_move_idx: int = 0
        self.last_board_state: Optional[List[List[int]]] = None
        self.last_tray_state: Optional[List[Tuple[Optional[str], Optional[Tuple[int, int]]]]] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_cleared_lines: int = 0
        self.current_combo_streak: int = 0
        self.rounds_played: int = 0
        self.total_pieces_placed: int = 0
        self.total_score: int = 0
        self.is_game_over: bool = False
        self.drop_retry_count: int = 0
        self.max_drop_retries: int = 2
        self.settle_delay: float = 0.40
        self.next_state_after_settle: BotState = BotState.CAPTURE_FRAME

    def _load_champion_network(self) -> Optional[neat.nn.FeedForwardNetwork]:
        """Loads champion network from best_champion.pkl."""
        return load_champion_network(CHAMPION_PATH, CONFIG_PATH)

    def capture_screen(self) -> np.ndarray:
        """Captures raw 1080x2400 screen frame via direct ADB socket or fallback."""
        return self.adb_client.screencap_cv2()

    def execute_swipe(
        self,
        start_x: Union[int, float],
        start_y: Union[int, float],
        end_x: Union[int, float],
        end_y: Union[int, float],
        duration_ms: Optional[int] = None,
    ) -> bool:
        """
        Dispatches a fluid touch swipe gesture with safe release coordinate clamping (Y <= 1580).
        """
        if duration_ms is None:
            dist = np.hypot(end_x - start_x, end_y - start_y)
            duration_ms = max(600, int(dist * 0.75))

        return self.adb_client.swipe(
            x1=start_x,
            y1=start_y,
            x2=end_x,
            y2=end_y,
            duration_ms=int(duration_ms),
            clamp_safe_release=True,
        )

    def plan_batch_moves(
        self,
        board_state: Union[np.ndarray, List[List[int]]],
        tray_pieces: Union[
            List[Dict[str, Any]],
            List[Optional[Piece]],
            List[Tuple[Optional[str], Optional[Tuple[int, int]]]],
        ],
        neat_net: Optional[neat.nn.FeedForwardNetwork] = None,
    ) -> List[Tuple[int, int, int]]:
        """Delegates to batch 3-piece permutation planner."""
        net_to_use = neat_net if neat_net is not None else self.net
        return plan_batch_moves(board_state, tray_pieces, net_to_use)

    def plan_moves_in_memory(
        self,
        real_board: List[List[int]],
        tray_pieces: List[Tuple[Optional[str], Optional[Tuple[int, int]]]],
    ) -> List[Tuple[int, int, int, Tuple[int, int], Tuple[int, int]]]:
        """
        Plans batch moves and computes exact screen touch gesture coordinates
        with safe release clamping Y <= 1580.
        
        Returns:
            List of (slot_idx, row, col, (start_x, start_y), (end_x, end_y))
        """
        optimal_moves = plan_batch_moves(real_board, tray_pieces, self.net)
        if not optimal_moves:
            return []

        planned_gestures: List[Tuple[int, int, int, Tuple[int, int], Tuple[int, int]]] = []

        for p_idx, r, c in optimal_moves:
            if not (0 <= p_idx < len(tray_pieces)):
                continue
            item = tray_pieces[p_idx]
            if item is None:
                continue

            p_name = None
            start_xy = None
            if isinstance(item, dict):
                p_name = item.get("name")
                start_xy = item.get("grab_xy")
            elif isinstance(item, (tuple, list)):
                p_name = item[0]
                start_xy = item[1] if len(item) > 1 else None
            elif isinstance(item, Piece):
                p_name = item.name
            elif isinstance(item, str):
                p_name = item

            if p_name is None or p_name not in BLOCK_SHAPES:
                continue

            if start_xy is None:
                scx = self.detector.slot_centers_x[p_idx] if hasattr(self.detector, "slot_centers_x") else (220.0 + p_idx * 320.0)
                start_xy = (int(scx), 1855)

            piece = Piece(p_name)
            start_x = clamp_coordinate_x(start_xy[0])
            start_y = clamp_coordinate_y(start_xy[1])

            # Centro da peça na grade do tabuleiro com offset vertical de 275px
            cw = self.detector.cell_w
            ch = self.detector.cell_h
            ex = self.detector.board_left + (c + (piece.width - 1) / 2.0) * cw + cw / 2.0
            ey = self.detector.board_top + (r + (piece.height - 1) / 2.0) * ch + ch / 2.0 + 275.0

            # Compensação do ganho do motor de física do Unity no Block Blast (DRAG_GAIN = 1.4x)
            # O jogo multiplica o deslocamento do dedo por 1.4x, logo o dedo deve se mover (alvo - início) / 1.4
            cx = int(round(start_x + (ex - start_x) / 1.4))
            cy = int(round(start_y + (ey - start_y) / 1.4))

            finger_drop_x = clamp_coordinate_x(cx)
            finger_drop_y = clamp_release_y(cy)

            planned_gestures.append(
                (p_idx, r, c, (start_x, start_y), (finger_drop_x, finger_drop_y))
            )

        return planned_gestures

    def verify_drop(
        self,
        previous_board: List[List[int]],
        current_frame: np.ndarray,
        expected_piece: Optional[Piece] = None,
        expected_pos: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Closed-loop drop verification: checks if the board state changed
        in accordance with the placed piece or line clears.
        """
        curr_board = self.detector.detect_board_state(current_frame)
        prev_occupied = sum(sum(row) for row in previous_board)
        curr_occupied = sum(sum(row) for row in curr_board)

        # 1. If cells increased by expected piece size, drop succeeded
        if expected_piece is not None:
            if curr_occupied >= prev_occupied + expected_piece.size:
                return True

        # 2. If lines were cleared, occupancy may decrease or stay equal, but board changed
        if curr_board != previous_board:
            return True

        # 3. If expected cell coordinates are now occupied
        if expected_piece is not None and expected_pos is not None:
            r0, c0 = expected_pos
            all_placed = True
            for dr, dc in expected_piece.blocks:
                r, c = r0 + dr, c0 + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if curr_board[r][c] != 1:
                        all_placed = False
                        break
            if all_placed:
                return True

        return False

    def get_adaptive_settling_delay(
        self,
        lines_cleared: int = 0,
        is_refill: bool = False,
        combo_streak: int = 0,
    ) -> float:
        """
        Calculates adaptive settling sleep delay:
        - Base move settling: 0.40s.
        - Single line clear: 0.55s.
        - High combo / multi-line clear: 0.85s - 1.30s (full particle dissipation).
        - Tray refill bounce: 0.85s.
        """
        if is_refill:
            return 0.85

        if lines_cleared > 0:
            if lines_cleared >= 3 or combo_streak >= 3:
                return 1.10
            elif lines_cleared == 2 or combo_streak >= 2:
                return 0.75
            else:
                return 0.55

        return 0.40

    def handle_game_over(self, frame: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Detects game-over screen, extracts score cleanly, and triggers restart if enabled.
        """
        is_over, score = self.detector.detect_game_over(frame)
        if not is_over:
            return False, None

        self.is_game_over = True
        extracted_score = score or getattr(self, "total_score", 0)
        rounds = getattr(self, "rounds_played", 0)
        pieces = getattr(self, "total_pieces_placed", 0)
        logger.info(f"=== [GAME OVER] Final Score: {extracted_score} | Rounds: {rounds} | Pieces: {pieces} ===")

        if getattr(self, "auto_restart", False):
            logger.info("[BOT] Auto-restarting game: tapping restart button...")
            # Tap restart button location (center 540, 1500)
            if hasattr(self, "adb_client"):
                self.adb_client.tap(540, 1500)
            time.sleep(1.5)
            # Reset internal stats for new session
            self.is_game_over = False
            self.current_combo_streak = 0
            self.last_board_state = None
            self.current_plan = []
            self.current_move_idx = 0

        return True, extracted_score

    def step_fsm(self) -> BotState:
        """
        Advances the Finite State Machine by one logical state transition.
        Returns the new state after execution.
        """
        # =================================================================
        # STATE: IDLE -> CAPTURE_FRAME
        # =================================================================
        if self.state == BotState.IDLE:
            logger.debug("[FSM] IDLE -> Initializing session")
            self.current_plan = []
            self.current_move_idx = 0
            self.state = BotState.CAPTURE_FRAME
            return self.state

        # =================================================================
        # STATE: CAPTURE_FRAME -> DETECT_STATE
        # =================================================================
        if self.state == BotState.CAPTURE_FRAME:
            logger.debug("[FSM] CAPTURE_FRAME -> Capturing screen")
            try:
                self.last_frame = self.capture_screen()
                self.state = BotState.DETECT_STATE
            except Exception as e:
                logger.error(f"[FSM] Capture failed ({e}), transitioning to RECOVER")
                self.state = BotState.RECOVER
            return self.state

        # =================================================================
        # STATE: DETECT_STATE -> PLAN_BATCH or CHECK_GAME_OVER
        # =================================================================
        if self.state == BotState.DETECT_STATE:
            logger.debug("[FSM] DETECT_STATE -> Parsing board and tray")
            if self.last_frame is None:
                self.state = BotState.CAPTURE_FRAME
                return self.state

            # Check for game-over screen first
            is_over, _ = self.detector.detect_game_over(self.last_frame)
            if is_over:
                self.state = BotState.CHECK_GAME_OVER
                return self.state

            self.last_board_state = self.detector.detect_board_state(self.last_frame)
            self.last_tray_state = self.detector.detect_tray_pieces(self.last_frame)

            valid_pieces = [p[0] for p in self.last_tray_state if p[0] is not None]
            if not valid_pieces:
                # Tray may be in the middle of refill animation
                logger.debug("[FSM] No pieces in tray, waiting for refill settling")
                self.settle_delay = 0.85
                self.next_state_after_settle = BotState.CAPTURE_FRAME
                self.state = BotState.SETTLE_ANIMATION
                return self.state

            self.state = BotState.PLAN_BATCH
            return self.state

        # =================================================================
        # STATE: PLAN_BATCH -> EXECUTE_MOVE or CHECK_GAME_OVER
        # =================================================================
        if self.state == BotState.PLAN_BATCH:
            logger.debug("[FSM] PLAN_BATCH -> Running NEAT 3-piece batch planner")
            if self.last_board_state is None or self.last_tray_state is None:
                self.state = BotState.CAPTURE_FRAME
                return self.state

            self.current_plan = self.plan_moves_in_memory(
                self.last_board_state, self.last_tray_state
            )
            self.current_move_idx = 0

            if not self.current_plan:
                logger.warning("[FSM] No valid moves found. Board full or game over!")
                self.state = BotState.CHECK_GAME_OVER
                return self.state

            logger.info(f"[FSM] Planned {len(self.current_plan)} batch moves successfully")
            self.state = BotState.EXECUTE_MOVE
            return self.state

        # =================================================================
        # STATE: EXECUTE_MOVE -> VERIFY_DROP
        # =================================================================
        if self.state == BotState.EXECUTE_MOVE:
            if self.current_move_idx >= len(self.current_plan):
                # All moves in current batch finished -> wait for tray refill
                self.rounds_played += 1
                self.settle_delay = self.get_adaptive_settling_delay(is_refill=True)
                self.next_state_after_settle = BotState.CAPTURE_FRAME
                self.state = BotState.SETTLE_ANIMATION
                return self.state

            move = self.current_plan[self.current_move_idx]
            p_idx, r, c, start_xy, end_xy = move
            p_name = self.last_tray_state[p_idx][0] if self.last_tray_state else "?"

            logger.info(
                f"[FSM] Move {self.current_move_idx + 1}/{len(self.current_plan)}: "
                f"{p_name} -> ({r}, {c}) | Swipe {start_xy} -> {end_xy}"
            )
            success = self.execute_swipe(start_xy[0], start_xy[1], end_xy[0], end_xy[1])
            if not success:
                logger.error("[FSM] Swipe execution failed, entering RECOVER")
                self.state = BotState.RECOVER
                return self.state

            self.state = BotState.VERIFY_DROP
            return self.state

        # =================================================================
        # STATE: VERIFY_DROP -> SETTLE_ANIMATION or RECOVER
        # =================================================================
        if self.state == BotState.VERIFY_DROP:
            logger.debug("[FSM] VERIFY_DROP -> Checking closed-loop drop confirmation")
            time.sleep(0.35)  # Drop physics settling time

            try:
                verify_frame = self.capture_screen()
                move = self.current_plan[self.current_move_idx]
                p_idx, r, c, _, _ = move
                p_name = self.last_tray_state[p_idx][0] if self.last_tray_state else None
                expected_piece = Piece(p_name) if p_name and p_name in BLOCK_SHAPES else None

                dropped = self.verify_drop(
                    previous_board=self.last_board_state or [[0] * 8 for _ in range(8)],
                    current_frame=verify_frame,
                    expected_piece=expected_piece,
                    expected_pos=(r, c),
                )

                if dropped:
                    self.total_pieces_placed += 1
                    self.drop_retry_count = 0
                    self.last_board_state = self.detector.detect_board_state(verify_frame)
                    self.current_move_idx += 1

                    # Adaptive animation timing
                    if self.current_move_idx < len(self.current_plan):
                        self.settle_delay = self.get_adaptive_settling_delay(
                            lines_cleared=self.last_cleared_lines,
                            is_refill=False,
                            combo_streak=self.current_combo_streak,
                        )
                        self.next_state_after_settle = BotState.EXECUTE_MOVE
                    else:
                        self.rounds_played += 1
                        self.settle_delay = self.get_adaptive_settling_delay(is_refill=True)
                        self.next_state_after_settle = BotState.CAPTURE_FRAME

                    self.state = BotState.SETTLE_ANIMATION
                else:
                    self.drop_retry_count += 1
                    logger.warning(
                        f"[FSM] Drop unconfirmed (retry {self.drop_retry_count}/{self.max_drop_retries})"
                    )
                    if self.drop_retry_count <= self.max_drop_retries:
                        # Retry current swipe with slightly increased duration
                        self.state = BotState.EXECUTE_MOVE
                    else:
                        logger.error("[FSM] Drop failed after retries, recovering from fresh capture")
                        self.state = BotState.RECOVER

            except Exception as e:
                logger.error(f"[FSM] Error during drop verification ({e})")
                self.state = BotState.RECOVER

            return self.state

        # =================================================================
        # STATE: SETTLE_ANIMATION -> (next_state_after_settle)
        # =================================================================
        if self.state == BotState.SETTLE_ANIMATION:
            logger.debug(f"[FSM] SETTLE_ANIMATION -> Sleeping {self.settle_delay:.2f}s")
            time.sleep(self.settle_delay)
            self.state = self.next_state_after_settle
            return self.state

        # =================================================================
        # STATE: CHECK_GAME_OVER -> STOPPED or CAPTURE_FRAME
        # =================================================================
        if self.state == BotState.CHECK_GAME_OVER:
            logger.info("[FSM] CHECK_GAME_OVER -> Processing game termination")
            if self.last_frame is not None:
                is_over, final_score = self.handle_game_over(self.last_frame)
                if is_over and self.auto_restart:
                    self.state = BotState.CAPTURE_FRAME
                    return self.state

            if not self.auto_restart:
                self.state = BotState.STOPPED
            else:
                self.state = BotState.RECOVER
            return self.state

        # =================================================================
        # STATE: RECOVER -> CAPTURE_FRAME
        # =================================================================
        if self.state == BotState.RECOVER:
            logger.info("[FSM] RECOVER -> Resetting transient queue and re-syncing")
            self.current_plan = []
            self.current_move_idx = 0
            self.drop_retry_count = 0
            time.sleep(0.60)
            self.state = BotState.CAPTURE_FRAME
            return self.state

        # =================================================================
        # STATE: STOPPED
        # =================================================================
        return BotState.STOPPED

    def play_single_tray(self) -> bool:
        """
        Executes one full tray round via batch planning and sequential execution.
        Returns False if game over or no valid moves exist.
        """
        t0 = time.time()
        frame = self.capture_screen()
        t_cap = (time.time() - t0) * 1000

        # Check for game over
        is_over, _ = self.detector.detect_game_over(frame)
        if is_over:
            self.handle_game_over(frame)
            return False

        t1 = time.time()
        board = self.detector.detect_board_state(frame)
        tray = self.detector.detect_tray_pieces(frame)
        t_det = (time.time() - t1) * 1000

        valid_pieces = [p[0] for p in tray if p[0] is not None]
        if not valid_pieces:
            logger.debug("[BOT] No pieces in tray, waiting...")
            return False

        logger.info(f"[FRAME] Cap: {t_cap:.0f}ms | Vision: {t_det:.0f}ms | Tray: {valid_pieces}")

        t2 = time.time()
        moves = self.plan_moves_in_memory(board, tray)
        t_plan = (time.time() - t2) * 1000

        if not moves:
            logger.warning("[BOT] No valid moves found!")
            return False

        logger.info(f"[IA] Planned {len(moves)} moves in {t_plan:.1f}ms:")

        for idx, (p_idx, r, c, start_pos, end_pos) in enumerate(moves):
            p_name = tray[p_idx][0]
            logger.info(f"  👉 Move {idx + 1}: {p_name} -> ({r}, {c}) | {start_pos} -> {end_pos}")
            self.execute_swipe(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
            time.sleep(0.40)

        # Refill settling delay
        time.sleep(0.85)
        self.rounds_played += 1
        return True

    def run_fsm(
        self,
        max_rounds: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Continuous live execution driven by the formal FSM.
        """
        logger.info("=" * 65)
        logger.info("   BLOCK BLAST AI - FSM AUTONOMOUS LIVE PLAY LOOP (MOTO G54)")
        logger.info("=" * 65)

        self.state = BotState.IDLE
        step_count = 0

        try:
            while self.state != BotState.STOPPED:
                step_count += 1
                self.step_fsm()

                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"[FSM] Reached max_steps limit ({max_steps}). Stopping.")
                    break

                if max_rounds is not None and self.rounds_played >= max_rounds:
                    logger.info(f"[FSM] Reached max_rounds limit ({max_rounds}). Stopping.")
                    break

        except KeyboardInterrupt:
            logger.info("\n[BOT] Paused by user via KeyboardInterrupt.")

    def run_infinite(self):
        """Continuous live play loop (backward-compatible wrapper)."""
        self.run_fsm()


# =====================================================================
# CLI ENTRY POINT FOR MANUAL DIAGNOSTICS & PROBING
# =====================================================================

if __name__ == "__main__":
    bot = BlockBlastMobileBot()

    if len(sys.argv) > 1 and sys.argv[1] == "--test-plan":
        frame = bot.capture_screen()
        b = bot.detector.detect_board_state(frame)
        t = bot.detector.detect_tray_pieces(frame)
        moves = bot.plan_moves_in_memory(b, t)
        print(f"\n[TEST PLAN] Pieces: {[p[0] for p in t]}")
        print(f"[TEST PLAN] Planned Moves ({len(moves)}):")
        for idx, (p_idx, r, c, start_pos, end_pos) in enumerate(moves):
            print(f"  {idx + 1}. {t[p_idx][0]} -> ({r}, {c}) | {start_pos} -> {end_pos}")

    elif len(sys.argv) > 1 and sys.argv[1] == "--test-one-move":
        frame = bot.capture_screen()
        b = bot.detector.detect_board_state(frame)
        t = bot.detector.detect_tray_pieces(frame)
        moves = bot.plan_moves_in_memory(b, t)
        if not moves:
            print("[ERROR] No valid moves!")
            sys.exit(1)
        p_idx, r, c, start_pos, end_pos = moves[0]
        p_name = t[p_idx][0]
        print(f"[EXECUTING] {p_name} from {start_pos} to ({r}, {c}) at {end_pos}...")
        bot.execute_swipe(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
        time.sleep(0.8)
        after_frame = bot.capture_screen()
        after_board = bot.detector.detect_board_state(after_frame)
        placed_count = sum(sum(row) for row in after_board)
        print(f"[RESULT] Board occupancy after move: {placed_count} cells")

    elif len(sys.argv) > 1 and sys.argv[1] == "--run-fsm":
        bot.run_fsm()

    else:
        bot.run_infinite()
