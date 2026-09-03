"""
game.py - Motor Oficial de Block Blast v4 (Completo e Otimizado).
- 42 peças oficiais completas (incluindo diagonais 2x2 e 3x3, Z/S verticais, L horizontal e Plus).
- Heurísticas de Aderência (Encaixe tipo Quebra-Cabeça) e Compacidade de Tabuleiro.
- Preservação do Centro e Áreas Contínuas (evita morrer para peças 3x3).
- 78 features (64 células + 14 heurísticas estratégicas).
"""

import random
from typing import List, Tuple, Optional, Dict, Any

# =====================================================================
# TODAS AS 42 PEÇAS OFICIAIS DO BLOCK BLAST MOBILE
# =====================================================================
BLOCK_SHAPES = {
    # 1. Ponto (1 bloco)
    "dot": [(0, 0)],

    # 2. Dominós (2 blocos)
    "line2_h": [(0, 0), (0, 1)],
    "line2_v": [(0, 0), (1, 0)],

    # 3. Triominós Retos (3 blocos)
    "line3_h": [(0, 0), (0, 1), (0, 2)],
    "line3_v": [(0, 0), (1, 0), (2, 0)],

    # 4. Cantos Pequenos (3 blocos, 2x2)
    "corner_tl": [(0, 0), (0, 1), (1, 0)],
    "corner_tr": [(0, 0), (0, 1), (1, 1)],
    "corner_bl": [(0, 0), (1, 0), (1, 1)],
    "corner_br": [(0, 1), (1, 0), (1, 1)],

    # 5. Diagonais (2 e 3 blocos - oficiais do mobile!)
    "diag2_down": [(0, 0), (1, 1)],
    "diag2_up": [(0, 1), (1, 0)],
    "diag3_down": [(0, 0), (1, 1), (2, 2)],
    "diag3_up": [(0, 2), (1, 1), (2, 0)],

    # 6. Quadrados
    "square2x2": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "square3x3": [(r, c) for r in range(3) for c in range(3)],

    # 7. Barras Longas
    "line4_h": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "line4_v": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "line5_h": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "line5_v": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],

    # 8. Peças T (4 rotações)
    "t_up": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "t_down": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "t_left": [(0, 1), (1, 0), (1, 1), (2, 1)],
    "t_right": [(0, 0), (1, 0), (1, 1), (2, 0)],

    # 9. Peças L Verticais (3x2, 4 blocos)
    "l_v_up_l": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "l_v_up_r": [(0, 1), (1, 1), (2, 1), (2, 0)],
    "l_v_down_l": [(0, 0), (0, 1), (1, 0), (2, 0)],
    "l_v_down_r": [(0, 0), (0, 1), (1, 1), (2, 1)],

    # 10. Peças L Horizontais (2x3, 4 blocos)
    "l_h_tl": [(0, 0), (1, 0), (0, 1), (0, 2)],
    "l_h_tr": [(0, 0), (0, 1), (0, 2), (1, 2)],
    "l_h_bl": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "l_h_br": [(0, 2), (1, 0), (1, 1), (1, 2)],

    # 11. Peças Z e S (Horizontais e Verticais)
    "z_h": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "s_h": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "z_v": [(0, 1), (1, 0), (1, 1), (2, 0)],
    "s_v": [(0, 0), (1, 0), (1, 1), (2, 1)],

    # 12. Cantos Grandes (3x3, 5 blocos)
    "big_corner_tl": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    "big_corner_tr": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
    "big_corner_bl": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    "big_corner_br": [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],

    # 13. Retângulos 2x3 e 3x2 (6 blocos)
    "rect2x3": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "rect3x2": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],

    # 14. Cruz / Plus (5 blocos, 3x3)
    "plus_cross": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
}

PIECE_NAMES = list(BLOCK_SHAPES.keys())

PIECE_COLORS = {
    1: (255, 215, 0),    # Amarelo Ouro (Dot)
    2: (0, 220, 255),    # Ciano (2 blocos)
    3: (50, 205, 50),    # Verde (3 blocos)
    4: (255, 105, 180),  # Rosa (4 blocos)
    5: (255, 69, 0),     # Laranja-Avermelhado (5 blocos)
    6: (147, 112, 219),  # Roxo Médio (6 blocos)
    9: (220, 20, 60),    # Carmesim (3x3)
}


class Piece:
    __slots__ = ('name', 'blocks', 'size', 'color', 'height', 'width')
    def __init__(self, name: str):
        self.name = name
        self.blocks: List[Tuple[int, int]] = BLOCK_SHAPES[name]
        self.size = len(self.blocks)
        self.color = PIECE_COLORS.get(self.size, (100, 180, 255))
        self.height = max(r for r, c in self.blocks) + 1
        self.width = max(c for r, c in self.blocks) + 1


class BlockBlast:
    GRID_SIZE = 8

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.board = [[0] * 8 for _ in range(8)]
        self.tray: List[Optional[Piece]] = [None, None, None]
        self.score = 0
        self.lines_cleared_total = 0
        self.blocks_placed_total = 0
        self.moves_count = 0
        self.combo_streak = 0
        self.max_combo = 0
        self.turns_without_clear = 0
        self.COMBO_TOLERANCE = 2  # Tolerância: 2 jogadas sem limpar antes de zerar o combo
        self.board_clears = 0
        self.multi_clears = 0
        self.game_over = False
        self.fitness = 0.0
        self.refill_tray()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.board = [[0] * 8 for _ in range(8)]
        self.tray = [None, None, None]
        self.score = 0
        self.lines_cleared_total = 0
        self.blocks_placed_total = 0
        self.moves_count = 0
        self.combo_streak = 0
        self.max_combo = 0
        self.turns_without_clear = 0
        self.COMBO_TOLERANCE = 2
        self.board_clears = 0
        self.multi_clears = 0
        self.game_over = False
        self.fitness = 0.0
        self.refill_tray()

    def refill_tray(self):
        if all(p is None for p in self.tray):
            for i in range(3):
                self.tray[i] = Piece(self.rng.choice(PIECE_NAMES))

    def _can_place_on(self, piece: Piece, board: list, start_row: int, start_col: int) -> bool:
        """Testa posicionamento em qualquer tabuleiro (real ou simulado)."""
        for dr, dc in piece.blocks:
            r = start_row + dr
            c = start_col + dc
            if r < 0 or r >= 8 or c < 0 or c >= 8:
                return False
            if board[r][c] != 0:
                return False
        return True

    def can_place(self, piece: Piece, start_row: int, start_col: int) -> bool:
        if piece is None:
            return False
        return self._can_place_on(piece, self.board, start_row, start_col)

    def get_valid_moves(self) -> List[Tuple[int, int, int]]:
        """Retorna todas as jogadas legais (piece_idx, row, col)."""
        moves = []
        for p_idx, piece in enumerate(self.tray):
            if piece is None:
                continue
            max_r = 9 - piece.height
            max_c = 9 - piece.width
            for r in range(max_r):
                for c in range(max_c):
                    if self.can_place(piece, r, c):
                        moves.append((p_idx, r, c))
        return moves

    def _has_any_valid_move(self) -> bool:
        """Versão rápida: retorna True assim que acha UMA jogada válida."""
        for p_idx, piece in enumerate(self.tray):
            if piece is None:
                continue
            max_r = 9 - piece.height
            max_c = 9 - piece.width
            for r in range(max_r):
                for c in range(max_c):
                    if self.can_place(piece, r, c):
                        return True
        return False

    def _piece_fits_anywhere(self, piece: Piece, board: list) -> bool:
        """Checa se uma peça cabe em QUALQUER posição do tabuleiro."""
        max_r = 9 - piece.height
        max_c = 9 - piece.width
        for r in range(max_r):
            for c in range(max_c):
                if self._can_place_on(piece, board, r, c):
                    return True
        return False

    def clear_full_lines(self) -> int:
        """Limpa linhas e colunas completas simultaneamente."""
        rows_to_clear = [r for r in range(8) if all(self.board[r][c] != 0 for c in range(8))]
        cols_to_clear = [c for c in range(8) if all(self.board[row_idx][c] != 0 for row_idx in range(8))]

        for r in rows_to_clear:
            for c in range(8):
                self.board[r][c] = 0

        for c in cols_to_clear:
            for r in range(8):
                self.board[r][c] = 0

        return len(rows_to_clear) + len(cols_to_clear)

    def simulate_move_features(self, piece_idx: int, target_row: int, target_col: int) -> List[float]:
        """
        79 features = 64 células do tabuleiro + 15 heurísticas estratégicas.
        Inclui Aderência (Encaixe), Compacidade e Preservação de Área Contínua.
        """
        piece = self.tray[piece_idx]
        if piece is None:
            return [0.0] * 79

        # Clona e posiciona
        temp_board = [row[:] for row in self.board]
        contacts = 0

        for dr, dc in piece.blocks:
            r = target_row + dr
            c = target_col + dc
            temp_board[r][c] = 1

            # Contato com as paredes do tabuleiro (hugging walls)
            if r == 0 or r == 7:
                contacts += 1
            if c == 0 or c == 7:
                contacts += 1

            # Contato com blocos pré-existentes (encaixe tipo quebra-cabeça)
            if r > 0 and self.board[r - 1][c] != 0: contacts += 1
            if r < 7 and self.board[r + 1][c] != 0: contacts += 1
            if c > 0 and self.board[r][c - 1] != 0: contacts += 1
            if c < 7 and self.board[r][c + 1] != 0: contacts += 1

        # Limpa linhas/colunas no tabuleiro temporário
        rows_to_clear = [r for r in range(8) if all(temp_board[r][c] != 0 for c in range(8))]
        cols_to_clear = [c for c in range(8) if all(temp_board[row_idx][c] != 0 for row_idx in range(8))]
        cleared = len(rows_to_clear) + len(cols_to_clear)

        for r in rows_to_clear:
            for c in range(8):
                temp_board[r][c] = 0
        for c in cols_to_clear:
            for r in range(8):
                temp_board[r][c] = 0

        # === PASS ÚNICO NO TABULEIRO ===
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
                    # Contar buracos cercados (espaço morto difícil de preencher)
                    nb = 0
                    if r > 0 and temp_board[r - 1][c] != 0: nb += 1
                    if r < 7 and temp_board[r + 1][c] != 0: nb += 1
                    if c > 0 and temp_board[r][c - 1] != 0: nb += 1
                    if c < 7 and temp_board[r][c + 1] != 0: nb += 1
                    if nb >= 3:
                        trapped_holes += 1

                # Rugosidade / Perímetro exposto (quanto menor, mais compacto)
                if r < 7 and temp_board[r + 1][c] != val:
                    perimeter += 1
                if c < 7 and temp_board[r][c + 1] != val:
                    perimeter += 1

        # =====================================================================
        # 14 HEURÍSTICAS ESTRATÉGICAS HARMONIOSAS (todas normalizadas em [0, 1])
        # =====================================================================

        # H1: Linhas limpas nesta jogada
        features.append(min(1.0, cleared / 8.0))

        # H2: Taxa de ocupação do tabuleiro
        features.append(occupied_count / 64.0)

        # H3: Espaços 3x3 livres (crucial para não morrer com quadrados grandes)
        can_fit_3x3 = 0
        for r in range(6):
            for c in range(6):
                if all(temp_board[r + dr][c + dc] == 0 for dr in range(3) for dc in range(3)):
                    can_fit_3x3 += 1
        features.append(min(1.0, can_fit_3x3 / 10.0))

        # H4: Espaços para barras 5x1 livres (horizontal e vertical)
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

        # H6: Sequência de Combo efetiva (leva em conta a tolerância de 2 jogadas)
        if cleared > 0:
            effective_combo = self.combo_streak + 1
            combo_status = 1.0  # Combo ativo e renovado nesta jogada!
        elif self.combo_streak > 0 and (self.turns_without_clear + 1) < self.COMBO_TOLERANCE:
            effective_combo = self.combo_streak
            combo_status = 0.5  # Em janela de tolerância segura (preparando o terreno)
        else:
            effective_combo = 0
            combo_status = 0.0  # Sem combo ativo
        features.append(min(1.0, effective_combo / 15.0))

        # H7: Jogabilidade futura (se as peças restantes na bandeja ainda cabem)
        remaining_pieces = [self.tray[i] for i in range(3) if i != piece_idx and self.tray[i] is not None]
        if remaining_pieces:
            playable = sum(1 for rp in remaining_pieces if self._piece_fits_anywhere(rp, temp_board))
            features.append(playable / len(remaining_pieces))
        else:
            features.append(1.0)

        # H8: Linhas quase completas (6+ blocos = oportunidade de fechar combo)
        near_complete = sum(1 for cnt in row_counts if cnt >= 6) + sum(1 for cnt in col_counts if cnt >= 6)
        features.append(min(1.0, near_complete / 6.0))

        # H9: Altura máxima das colunas
        max_height = max(col_heights)
        features.append(max_height / 8.0)

        # H10: Variância de altura (estabilidade da superfície)
        avg_h = sum(col_heights) / 8.0
        variance = sum((h - avg_h) ** 2 for h in col_heights) / 8.0
        features.append(min(1.0, variance / 8.0))

        # H11: Zona de perigo (quando o tabuleiro ultrapassa 50% de ocupação)
        occupancy_ratio = occupied_count / 64.0
        danger = max(0.0, (occupancy_ratio - 0.5) * 2.0)
        features.append(danger)

        # H12: ADERÊNCIA / ENCAIXE (Contact Ratio - combate o jogo bagunçado)
        # Recompensa peças coladas a paredes ou blocos existentes
        max_contacts = max(1, piece.size * 4)
        contact_ratio = min(1.0, contacts / max_contacts)
        features.append(contact_ratio)

        # H13: COMPACIDADE / SUAVIDADE (Low Perimeter - favorece blocos agrupados)
        smoothness = max(0.0, 1.0 - (perimeter / 56.0))
        features.append(smoothness)

        # H14: PRESERVAÇÃO DE ESPAÇO CENTRAL LIVRE (Centro desimpedido para 3x3)
        center_openness = 1.0 - (center_occupied / 16.0)
        features.append(center_openness)

        # H15: STATUS DO COMBO SUSTENTÁVEL (1.0 = ativo/renovado, 0.5 = em tolerância, 0.0 = sem combo)
        features.append(combo_status)

        return features

    def step(self, piece_idx: int, target_row: int, target_col: int) -> Tuple[bool, int, float]:
        if self.game_over or not (0 <= piece_idx < 3) or self.tray[piece_idx] is None:
            self.game_over = True
            return False, 0, 0.0

        piece = self.tray[piece_idx]
        contacts = 0

        # 1. Medir contatos com paredes e blocos pré-existentes (encaixe tipo quebra-cabeça)
        for dr, dc in piece.blocks:
            r = target_row + dr
            c = target_col + dc
            if r == 0 or r == 7: contacts += 1
            if c == 0 or c == 7: contacts += 1
            if r > 0 and self.board[r - 1][c] != 0: contacts += 1
            if r < 7 and self.board[r + 1][c] != 0: contacts += 1
            if c > 0 and self.board[r][c - 1] != 0: contacts += 1
            if c < 7 and self.board[r][c + 1] != 0: contacts += 1

        # 2. Posiciona os blocos da peça no tabuleiro
        for dr, dc in piece.blocks:
            self.board[target_row + dr][target_col + dc] = 1

        self.tray[piece_idx] = None
        self.moves_count += 1
        self.blocks_placed_total += piece.size

        points = piece.size
        cleared_lines = self.clear_full_lines()

        # ========== RECOMPENSAS CALIBRADAS E EQUILIBRADAS ==========
        # Base: pequeno bônus por sobreviver (+1 por jogada)
        step_fitness = 1.0

        # Bônus suave de encaixe/aderência (máx +3): ensina a organizar sem atrapalhar os combos
        max_contacts = max(1, piece.size * 4)
        fit_ratio = min(1.0, contacts / max_contacts)
        step_fitness += fit_ratio * 3.0

        if cleared_lines > 0:
            self.combo_streak += 1
            self.turns_without_clear = 0
            self.lines_cleared_total += cleared_lines

            # =========================================================
            # PONTUAÇÃO OFICIAL BLOCK BLAST MOBILE
            # =========================================================
            # 1. Base triangular oficial: 10 (1 lin), 30 (2 lin), 60 (3 lin), 100 (4 lin)...
            line_base_score = 10 * (cleared_lines * (cleared_lines + 1)) // 2

            # 2. Bônus oficial de Combo por nível (só paga bônus do nível 2 em diante)
            combo_score = max(0, self.combo_streak - 1) * 10 * cleared_lines
            points += line_base_score + combo_score

            if self.combo_streak > self.max_combo:
                self.max_combo = self.combo_streak

            # =========================================================
            # FITNESS ESTRATÉGICO DE LONGO PRAZO
            # =========================================================
            combo_fitness = (cleared_lines * 50.0 * self.combo_streak + (self.combo_streak ** 2) * 20.0)
            step_fitness += combo_fitness

            # Limpeza múltipla (2+ linhas simultâneas)
            if cleared_lines >= 2:
                self.multi_clears += 1
                step_fitness += (cleared_lines ** 2) * 50.0

            # Board Clear total (tabuleiro 100% limpo: +300 pts no placar oficial!)
            if all(self.board[r][c] == 0 for r in range(8) for c in range(8)):
                self.board_clears += 1
                points += 300
                step_fitness += 3000.0
        else:
            self.turns_without_clear += 1
            # Tolerância de 2 jogadas sem limpar antes de resetar o combo
            if self.turns_without_clear >= self.COMBO_TOLERANCE:
                self.combo_streak = 0

        self.score += points
        self.refill_tray()

        # Verifica se ainda há movimentos legais
        if not self._has_any_valid_move():
            self.game_over = True
            # ANTI-SUICÍDIO: Subtrai do fitness severamente, em vez de apagar os méritos da jogada!
            step_fitness -= 100.0

        self.fitness += step_fitness
        return True, points, step_fitness

    def clone(self) -> 'BlockBlast':
        """Creates a complete independent deep copy of the current game state."""
        cloned = BlockBlast.__new__(BlockBlast)
        cloned.rng = random.Random()
        cloned.rng.setstate(self.rng.getstate())
        cloned.board = [row[:] for row in self.board]
        cloned.tray = [Piece(p.name) if p is not None else None for p in self.tray]
        cloned.score = self.score
        cloned.lines_cleared_total = self.lines_cleared_total
        cloned.blocks_placed_total = self.blocks_placed_total
        cloned.moves_count = self.moves_count
        cloned.combo_streak = self.combo_streak
        cloned.max_combo = self.max_combo
        cloned.turns_without_clear = self.turns_without_clear
        cloned.COMBO_TOLERANCE = self.COMBO_TOLERANCE
        cloned.board_clears = self.board_clears
        cloned.multi_clears = self.multi_clears
        cloned.game_over = self.game_over
        cloned.fitness = self.fitness
        return cloned

    def get_board_copy(self) -> List[List[int]]:
        """Returns a 2D copy of the current 8x8 board matrix."""
        return [row[:] for row in self.board]

    def get_valid_moves_for_piece(self, piece_idx: int) -> List[Tuple[int, int, int]]:
        """Returns all legal (piece_idx, row, col) placements for a specific tray slot."""
        if not (0 <= piece_idx < 3) or self.tray[piece_idx] is None:
            return []
        piece = self.tray[piece_idx]
        max_r = 9 - piece.height
        max_c = 9 - piece.width
        moves = []
        for r in range(max_r):
            for c in range(max_c):
                if self.can_place(piece, r, c):
                    moves.append((piece_idx, r, c))
        return moves

    def simulate_batch(self, moves: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Simulates a batch sequence of moves on this game instance's current board."""
        return simulate_batch_sequence(
            board=self.board,
            pieces=self.tray,
            moves=moves,
            initial_combo=self.combo_streak,
            initial_turns_without_clear=self.turns_without_clear,
            combo_tolerance=self.COMBO_TOLERANCE,
        )


def simulate_batch_sequence(
    board: List[List[int]],
    pieces: List[Optional[Piece]],
    moves: List[Tuple[int, int, int]],
    initial_combo: int = 0,
    initial_turns_without_clear: int = 0,
    combo_tolerance: int = 2,
) -> Dict[str, Any]:
    """
    Simulates placing a batch sequence of pieces on an 8x8 board in-memory.
    
    Args:
        board: 8x8 list of lists (0=empty, 1=occupied).
        pieces: list of Piece objects (or None for empty slots).
        moves: list of (piece_idx, target_row, target_col).
        initial_combo: starting combo streak.
        initial_turns_without_clear: starting turns without clearing lines.
        combo_tolerance: tolerance turns before combo resets.
        
    Returns:
        Dict with:
            - 'valid': bool (True if all moves were legal)
            - 'final_board': List[List[int]] (8x8 updated board)
            - 'total_score': int (total official score accumulated)
            - 'total_fitness': float (total strategic fitness)
            - 'lines_cleared_total': int (total cleared lines)
            - 'final_combo': int (combo streak at the end)
            - 'max_combo': int (highest combo reached)
            - 'moves_executed': int (count of valid moves executed)
            - 'board_cleared': bool (True if board is 100% empty at end)
            - 'failed_step': Optional[int] (index of first failed move if any)
            - 'step_results': List[Dict] (step-by-step telemetry)
    """
    curr_board = [row[:] for row in board]
    curr_pieces = list(pieces)
    score = 0
    fitness = 0.0
    lines_total = 0
    combo_streak = initial_combo
    max_combo = initial_combo
    turns_without_clear = initial_turns_without_clear
    step_results = []
    
    for step_idx, (p_idx, target_r, target_c) in enumerate(moves):
        if not (0 <= p_idx < len(curr_pieces)) or curr_pieces[p_idx] is None:
            return {
                "valid": False,
                "final_board": curr_board,
                "total_score": score,
                "total_fitness": fitness,
                "lines_cleared_total": lines_total,
                "final_combo": combo_streak,
                "max_combo": max_combo,
                "moves_executed": step_idx,
                "board_cleared": all(curr_board[r][c] == 0 for r in range(8) for c in range(8)),
                "failed_step": step_idx,
                "step_results": step_results,
            }
        
        piece = curr_pieces[p_idx]
        
        # Check placement validity
        can_place = True
        for dr, dc in piece.blocks:
            r = target_r + dr
            c = target_c + dc
            if r < 0 or r >= 8 or c < 0 or c >= 8 or curr_board[r][c] != 0:
                can_place = False
                break
        
        if not can_place:
            return {
                "valid": False,
                "final_board": curr_board,
                "total_score": score,
                "total_fitness": fitness,
                "lines_cleared_total": lines_total,
                "final_combo": combo_streak,
                "max_combo": max_combo,
                "moves_executed": step_idx,
                "board_cleared": all(curr_board[r][c] == 0 for r in range(8) for c in range(8)),
                "failed_step": step_idx,
                "step_results": step_results,
            }
        
        # Contacts calculation
        contacts = 0
        for dr, dc in piece.blocks:
            r = target_r + dr
            c = target_c + dc
            if r == 0 or r == 7: contacts += 1
            if c == 0 or c == 7: contacts += 1
            if r > 0 and curr_board[r - 1][c] != 0: contacts += 1
            if r < 7 and curr_board[r + 1][c] != 0: contacts += 1
            if c > 0 and curr_board[r][c - 1] != 0: contacts += 1
            if c < 7 and curr_board[r][c + 1] != 0: contacts += 1
        
        # Place blocks
        for dr, dc in piece.blocks:
            curr_board[target_r + dr][target_c + dc] = 1
        
        curr_pieces[p_idx] = None
        step_pts = piece.size
        
        # Clear lines
        rows_to_clear = [r for r in range(8) if all(curr_board[r][c] != 0 for c in range(8))]
        cols_to_clear = [c for c in range(8) if all(curr_board[row_idx][c] != 0 for row_idx in range(8))]
        cleared_lines = len(rows_to_clear) + len(cols_to_clear)
        
        for r in rows_to_clear:
            for c in range(8):
                curr_board[r][c] = 0
        for c in cols_to_clear:
            for r in range(8):
                curr_board[r][c] = 0
        
        step_fit = 1.0
        max_contacts = max(1, piece.size * 4)
        fit_ratio = min(1.0, contacts / max_contacts)
        step_fit += fit_ratio * 3.0
        
        if cleared_lines > 0:
            combo_streak += 1
            turns_without_clear = 0
            lines_total += cleared_lines
            line_base_score = 10 * (cleared_lines * (cleared_lines + 1)) // 2
            combo_score = max(0, combo_streak - 1) * 10 * cleared_lines
            step_pts += line_base_score + combo_score
            if combo_streak > max_combo:
                max_combo = combo_streak
            
            step_fit += (cleared_lines * 50.0 * combo_streak + (combo_streak ** 2) * 20.0)
            if cleared_lines >= 2:
                step_fit += (cleared_lines ** 2) * 50.0
            
            if all(curr_board[r][c] == 0 for r in range(8) for c in range(8)):
                step_pts += 300
                step_fit += 3000.0
        else:
            turns_without_clear += 1
            if turns_without_clear >= combo_tolerance:
                combo_streak = 0
        
        score += step_pts
        fitness += step_fit
        step_results.append({
            "step_index": step_idx,
            "piece_idx": p_idx,
            "target_row": target_r,
            "target_col": target_c,
            "points": step_pts,
            "fitness": step_fit,
            "lines_cleared": cleared_lines,
            "combo_streak": combo_streak,
        })
    
    return {
        "valid": True,
        "final_board": curr_board,
        "total_score": score,
        "total_fitness": fitness,
        "lines_cleared_total": lines_total,
        "final_combo": combo_streak,
        "max_combo": max_combo,
        "moves_executed": len(moves),
        "board_cleared": all(curr_board[r][c] == 0 for r in range(8) for c in range(8)),
        "failed_step": None,
        "step_results": step_results,
    }
