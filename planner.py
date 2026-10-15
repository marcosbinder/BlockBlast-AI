"""
planner.py - Canonical Single Source of Truth for Batch 3-Piece Lookahead Planning & Tactical State Evaluation.

Provides:
- In-memory sequential simulation with immediate full row/column clearing after each piece.
- Dynamic combo streak tracking and official triangular line clear scoring.
- Combinatorial 3-piece permutation lookahead (3! = 6 sequences) using beam search.
- Canonical 79-feature vector extraction (64 grid cells + 15 normalized strategic heuristics).
- Topological board health evaluation (anti-hole penalty, 3x3 center preservation, orphan block minimization).
- Full compatibility with pre-trained NEAT champion neural network (checkpoints/best_champion.pkl).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import neat
import numpy as np

from game import BLOCK_SHAPES, Piece


def eval_board_tactics(board: List[List[int]]) -> float:
    """
    Evaluates the topological health and viability of an 8x8 board state:
    - Penalizes orphan blocks (isolated single blocks with 0 orthogonal neighbors).
    - Penalizes trapped holes (dead-end cavities surrounded on >= 3 sides).
    - Rewards open contiguous 3x3 areas to prevent game-over on large pieces.
    - Rewards center openness (rows 2-5, cols 2-5).
    """
    orphans = 0
    holes = 0
    center_occupied = 0

    for r in range(8):
        row = board[r]
        for c in range(8):
            if row[c] != 0:
                nb = 0
                if r > 0 and board[r - 1][c] != 0:
                    nb += 1
                if r < 7 and board[r + 1][c] != 0:
                    nb += 1
                if c > 0 and row[c - 1] != 0:
                    nb += 1
                if c < 7 and row[c + 1] != 0:
                    nb += 1
                if nb == 0:
                    orphans += 1
                if 2 <= r <= 5 and 2 <= c <= 5:
                    center_occupied += 1
            else:
                nb = 0
                if r > 0 and board[r - 1][c] != 0:
                    nb += 1
                if r < 7 and board[r + 1][c] != 0:
                    nb += 1
                if c > 0 and row[c - 1] != 0:
                    nb += 1
                if c < 7 and row[c + 1] != 0:
                    nb += 1
                if nb >= 3:
                    holes += 1

    can_fit_3x3 = False
    for r in range(6):
        r0 = board[r]
        r1 = board[r + 1]
        r2 = board[r + 2]
        for c in range(6):
            if (
                r0[c] == 0 and r0[c + 1] == 0 and r0[c + 2] == 0
                and r1[c] == 0 and r1[c + 1] == 0 and r1[c + 2] == 0
                and r2[c] == 0 and r2[c + 1] == 0 and r2[c + 2] == 0
            ):
                can_fit_3x3 = True
                break
        if can_fit_3x3:
            break

    tactic_score = 0.0
    tactic_score -= orphans * 4.0
    tactic_score -= min(12.0, holes * 3.0)
    tactic_score += 6.0 if can_fit_3x3 else -10.0
    center_openness = (16 - center_occupied) / 16.0
    tactic_score += center_openness * 3.0
    return tactic_score


def fast_simulate_placement(
    board: List[List[int]],
    piece: Piece,
    r: int,
    c: int,
) -> Optional[Tuple[List[List[int]], int, int, int]]:
    """
    Simulates placing a piece on an 8x8 board in-memory.
    Performs bounds/collision check, block placement, contact ratio calculation,
    and immediate full row/column clearing across all 8 rows and 8 columns.

    Returns:
        (new_board, points, cleared_lines_count, contacts) or None if placement is illegal.
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
        if pr == 0 or pr == 7:
            contacts += 1
        if pc == 0 or pc == 7:
            contacts += 1
        if pr > 0 and board[pr - 1][pc] != 0:
            contacts += 1
        if pr < 7 and board[pr + 1][pc] != 0:
            contacts += 1
        if pc > 0 and board[pr][pc - 1] != 0:
            contacts += 1
        if pc < 7 and board[pr][pc + 1] != 0:
            contacts += 1

    # 3. Line clearing across all 8 rows and 8 columns simultaneously
    rows_to_clear = [ri for ri in range(8) if all(new_board[ri][ci] != 0 for ci in range(8))]
    cols_to_clear = [ci for ci in range(8) if all(new_board[ri][ci] != 0 for ri in range(8))]
    cleared = len(rows_to_clear) + len(cols_to_clear)

    for ri in rows_to_clear:
        new_board[ri] = [0] * 8
    for ci in cols_to_clear:
        for ri in range(8):
            new_board[ri][ci] = 0

    points = piece.size
    if cleared > 0:
        points += 10 * (cleared * (cleared + 1)) // 2
        if all(new_board[row_idx][col_idx] == 0 for row_idx in range(8) for col_idx in range(8)):
            points += 300

    return new_board, points, cleared, contacts


def extract_move_features(
    board: List[List[int]],
    piece: Optional[Piece],
    target_row: int,
    target_col: int,
    current_combo: int = 0,
    turns_without_clear: int = 0,
    combo_tolerance: int = 2,
    remaining_pieces: Optional[List[Piece]] = None,
) -> List[float]:
    """
    Extracts the canonical 79-feature vector (64 cells + 15 strategic heuristics)
    for a proposed move on the given board state.
    Matches checkpoints/best_champion.pkl and config-feedforward topology.
    """
    if piece is None:
        return [0.0] * 79

    # Clone and place
    temp_board = [row[:] for row in board]
    contacts = 0

    for dr, dc in piece.blocks:
        r = target_row + dr
        c = target_col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            temp_board[r][c] = 1
            if r == 0 or r == 7:
                contacts += 1
            if c == 0 or c == 7:
                contacts += 1
            if r > 0 and board[r - 1][c] != 0:
                contacts += 1
            if r < 7 and board[r + 1][c] != 0:
                contacts += 1
            if c > 0 and board[r][c - 1] != 0:
                contacts += 1
            if c < 7 and board[r][c + 1] != 0:
                contacts += 1

    # Clear lines simultaneously
    rows_to_clear = [r for r in range(8) if all(temp_board[r][c] != 0 for c in range(8))]
    cols_to_clear = [c for c in range(8) if all(temp_board[ri][c] != 0 for ri in range(8))]
    cleared = len(rows_to_clear) + len(cols_to_clear)

    for r in rows_to_clear:
        temp_board[r] = [0] * 8
    for c in cols_to_clear:
        for r in range(8):
            temp_board[r][c] = 0

    # Single pass feature aggregation
    features = [0.0] * 64
    occupied_count = 0
    col_heights = [0] * 8
    row_counts = [0] * 8
    col_counts = [0] * 8
    trapped_holes = 0
    perimeter = 0
    center_occupied = 0

    for r in range(8):
        for c in range(8):
            val = temp_board[r][c]
            if val != 0:
                features[r * 8 + c] = 1.0
                occupied_count += 1
                row_counts[r] += 1
                col_counts[c] += 1
                if col_heights[c] == 0:
                    col_heights[c] = 8 - r
                if 2 <= r <= 5 and 2 <= c <= 5:
                    center_occupied += 1
            else:
                nb = 0
                if r > 0 and temp_board[r - 1][c] != 0:
                    nb += 1
                if r < 7 and temp_board[r + 1][c] != 0:
                    nb += 1
                if c > 0 and temp_board[r][c - 1] != 0:
                    nb += 1
                if c < 7 and temp_board[r][c + 1] != 0:
                    nb += 1
                if nb >= 3:
                    trapped_holes += 1

            if r < 7 and temp_board[r + 1][c] != val:
                perimeter += 1
            if c < 7 and temp_board[r][c + 1] != val:
                perimeter += 1

    # 15 Strategic Heuristics
    # H1: Linhas limpas
    features.append(min(1.0, cleared / 8.0))

    # H2: Taxa de ocupação
    features.append(occupied_count / 64.0)

    # H3: Espaços 3x3 livres
    can_fit_3x3 = 0
    for r in range(6):
        for c in range(6):
            if all(temp_board[r + dr][c + dc] == 0 for dr in range(3) for dc in range(3)):
                can_fit_3x3 += 1
    features.append(min(1.0, can_fit_3x3 / 10.0))

    # H4: Espaços para barras 5x1 livres
    can_fit_bar = 0
    for r in range(8):
        for c in range(4):
            if all(temp_board[r][c + dc] == 0 for dc in range(5)):
                can_fit_bar += 1
    for c in range(8):
        for r in range(4):
            if all(temp_board[r + dr][c] == 0 for dr in range(5)):
                can_fit_bar += 1
    features.append(min(1.0, can_fit_bar / 10.0))

    # H5: Penalidade de buracos mortais
    features.append(min(1.0, trapped_holes / 10.0))

    # H6: Sequência de Combo efetiva
    if cleared > 0:
        effective_combo = current_combo + 1
        combo_status = 1.0
    elif current_combo > 0 and (turns_without_clear + 1) < combo_tolerance:
        effective_combo = current_combo
        combo_status = 0.5
    else:
        effective_combo = 0
        combo_status = 0.0
    features.append(min(1.0, effective_combo / 15.0))

    # H7: Jogabilidade futura
    if remaining_pieces:
        playable = 0
        for rp in remaining_pieces:
            fits = False
            mr = 9 - rp.height
            mc = 9 - rp.width
            for r in range(mr):
                for c in range(mc):
                    if all(temp_board[r + dr][c + dc] == 0 for dr, dc in rp.blocks):
                        fits = True
                        break
                if fits:
                    break
            if fits:
                playable += 1
        features.append(playable / len(remaining_pieces))
    else:
        features.append(1.0)

    # H8: Linhas quase completas (6+ blocos)
    near_complete = sum(1 for cnt in row_counts if cnt >= 6) + sum(1 for cnt in col_counts if cnt >= 6)
    features.append(min(1.0, near_complete / 6.0))

    # H9: Altura máxima
    max_height = max(col_heights)
    features.append(max_height / 8.0)

    # H10: Variância de altura
    avg_h = sum(col_heights) / 8.0
    variance = sum((h - avg_h) ** 2 for h in col_heights) / 8.0
    features.append(min(1.0, variance / 8.0))

    # H11: Zona de perigo
    danger = max(0.0, (occupied_count / 64.0 - 0.5) * 2.0)
    features.append(danger)

    # H12: Aderência / Encaixe
    max_contacts = max(1, piece.size * 4)
    contact_ratio = min(1.0, contacts / max_contacts)
    features.append(contact_ratio)

    # H13: Compacidade (baixa rugosidade)
    smoothness = max(0.0, 1.0 - (perimeter / 56.0))
    features.append(smoothness)

    # H14: Preservação do centro livre
    center_openness = 1.0 - (center_occupied / 16.0)
    features.append(center_openness)

    # H15: Status do combo
    features.append(combo_status)

    return features


def plan_batch_moves(
    board_state: Union[np.ndarray, List[List[int]]],
    tray_pieces: Union[
        List[Dict[str, Any]],
        List[Optional[Piece]],
        List[Tuple[Optional[str], Optional[Tuple[int, int]]]],
        Sequence[Any],
    ],
    neat_net: Optional[neat.nn.FeedForwardNetwork] = None,
    beam_width: int = 8,
) -> List[Tuple[int, int, int]]:
    """
    Canonical 3-piece permutation beam search planner.
    Evaluates up to 3! = 6 sequence permutations in-memory, updating intermediate
    boards with exact line clearing and combo tracking between moves.

    Returns:
        List of (slot_idx, row, col) tuples in optimal execution order.
    """
    # 1. Normalize board
    if isinstance(board_state, np.ndarray):
        board = board_state.astype(int).tolist()
    else:
        board = [list(row) for row in board_state]

    # 2. Normalize tray pieces mapping slot_idx -> Piece
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

    # Single piece fast evaluation
    if len(active_slots) == 1:
        s0 = active_slots[0]
        piece0 = slot_pieces[s0]
        max_r = 9 - piece0.height
        max_c = 9 - piece0.width
        valid_moves = []
        for r in range(max_r):
            for c in range(max_c):
                sim_res = fast_simulate_placement(board, piece0, r, c)
                if sim_res is not None:
                    valid_moves.append((r, c, sim_res))
        if not valid_moves:
            return []

        best_m = (s0, valid_moves[0][0], valid_moves[0][1])
        best_score = -float("inf")
        for r, c, (nb, pts, clr, contacts) in valid_moves:
            tactics = eval_board_tactics(nb)
            if neat_net is not None:
                feats = extract_move_features(board, piece0, r, c)
                net_val = neat_net.activate(feats)[0]
            else:
                net_val = 0.0
            score = pts + clr * 50.0 + tactics + net_val * 20.0
            if score > best_score:
                best_score = score
                best_m = (s0, r, c)
        return [best_m]

    # Evaluate sequence permutations (up to 3! = 6 paths)
    permutations = list(itertools.permutations(active_slots))
    candidate_sequences: List[Tuple[float, List[Tuple[int, int, int]], List[List[int]]]] = []

    for perm in permutations:
        k = len(perm)
        current_beams: List[Tuple[float, List[List[int]], List[Tuple[int, int, int]], int]] = [
            (0.0, [row[:] for row in board], [], 0)
        ]

        for depth, slot_idx in enumerate(perm):
            piece = slot_pieces[slot_idx]
            next_beams = []
            max_r = 9 - piece.height
            max_c = 9 - piece.width

            for acc_score, b_state, path, combo in current_beams:
                candidates = []
                for r in range(max_r):
                    for c in range(max_c):
                        sim_res = fast_simulate_placement(b_state, piece, r, c)
                        if sim_res is None:
                            continue
                        new_b, pts, cleared, contacts = sim_res
                        new_combo = combo + 1 if cleared > 0 else 0
                        max_contacts = max(1, piece.size * 4)
                        contact_fit = (contacts / max_contacts) * 3.0
                        combo_bonus = cleared * 50.0 * (new_combo if new_combo > 0 else 1.0)
                        step_score = pts + combo_bonus + contact_fit
                        candidates.append((step_score, new_b, path + [(slot_idx, r, c)], new_combo))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for c_val, c_board, c_path, c_combo in candidates[:beam_width]:
                        next_beams.append((acc_score + c_val, c_board, c_path, c_combo))
                else:
                    # Dead-end for this branch: keep current valid prefix as candidate
                    if path:
                        deadlock_penalty = -300.0 * (k - depth)
                        tactics = eval_board_tactics(b_state)
                        candidate_sequences.append((acc_score + deadlock_penalty + tactics, path, b_state))

            if next_beams:
                next_beams.sort(key=lambda x: x[0], reverse=True)
                current_beams = next_beams[:beam_width]
            else:
                break

        for acc_score, b_state, path, _ in current_beams:
            if path:
                if len(path) == k:
                    acc_score += 500.0  # Full sequence completion bonus
                tactics = eval_board_tactics(b_state)
                candidate_sequences.append((acc_score + tactics, path, b_state))

    if not candidate_sequences:
        return []

    # Sort candidates by:
    # 1. Sequence length (prefer placing more pieces)
    # 2. Cumulative path score
    candidate_sequences.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

    # Re-evaluate top candidates with NEAT champion network if available
    top_candidates = candidate_sequences[:6]
    best_path = top_candidates[0][1]
    best_final_score = -float("inf")

    for base_score, path, final_board in top_candidates:
        if not path:
            continue
        final_score = base_score
        if neat_net is not None and len(path) == len(active_slots):
            # Final board feature evaluation
            dummy_piece = Piece("dot")
            feats = extract_move_features(final_board, dummy_piece, 0, 0)
            net_val = neat_net.activate(feats)[0]
            final_score += net_val * 20.0

        if final_score > best_final_score:
            best_final_score = final_score
            best_path = path

    return best_path


def choose_best_move(
    game_or_board: Any,
    tray_or_net: Any,
    neat_net: Optional[neat.nn.FeedForwardNetwork] = None,
) -> Optional[Tuple[int, int, int]]:
    """
    Unified canonical move selector.
    Supports either:
        choose_best_move(game, net) -> Tuple[int, int, int]
    or
        choose_best_move(board, tray, net) -> Tuple[int, int, int]
    Uses plan_batch_moves to evaluate optimal sequence lookahead.
    """
    # Pattern 1: (game, net)
    if hasattr(game_or_board, "board") and hasattr(game_or_board, "tray"):
        game = game_or_board
        net = tray_or_net
        tray_list = [(p.name, (0, 0)) if p is not None else (None, None) for p in game.tray]
        plan = plan_batch_moves(game.board, tray_list, net)
        if plan:
            return plan[0]

        # Fallback greedy
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        best_val = -float("inf")
        best_m = valid_moves[0]
        for p_idx, r, c in valid_moves:
            feats = game.simulate_move_features(p_idx, r, c)
            val = net.activate(feats)[0] if net is not None else 0.0
            if val > best_val:
                best_val = val
                best_m = (p_idx, r, c)
        return best_m

    # Pattern 2: (board, tray, net)
    board = game_or_board
    tray = tray_or_net
    net = neat_net
    plan = plan_batch_moves(board, tray, net)
    return plan[0] if plan else None
