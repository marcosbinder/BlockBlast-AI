"""
replay.py - Visualizador do Campeão com Avaliação Inteligente de Jogadas e Validação de Modelo.
"""

import os
import sys
import pickle
import random
import neat
import pygame
from typing import Optional, Tuple

from game import BlockBlast, Piece

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_champion.pkl")

# Paleta de Cores
COLOR_BG = (14, 16, 24)
COLOR_CARD_BG = (22, 25, 38)
COLOR_CARD_BORDER = (40, 46, 68)
COLOR_GRID_EMPTY = (30, 34, 50)
COLOR_GRID_BORDER = (42, 48, 70)
COLOR_TEXT_MAIN = (245, 245, 255)
COLOR_TEXT_MUTED = (140, 150, 175)
COLOR_CYAN = (0, 230, 200)
COLOR_GOLD = (255, 205, 50)
COLOR_BLUE = (65, 135, 245)
COLOR_BLUE_LIGHT = (130, 180, 255)
COLOR_GREEN = (45, 215, 120)


class MatchViewer:
    def __init__(self, width: int = 500, height: int = 680):
        pygame.init()
        pygame.display.set_caption("Block Blast AI - Replay do Campeão")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Segoe UI", 16, bold=True) if pygame.font.match_font("Segoe UI") else pygame.font.Font(None, 22)
        self.font_main = pygame.font.SysFont("Segoe UI", 13) if pygame.font.match_font("Segoe UI") else pygame.font.Font(None, 18)
        self.font_small = pygame.font.SysFont("Segoe UI", 11) if pygame.font.match_font("Segoe UI") else pygame.font.Font(None, 15)

        self.board_rect = pygame.Rect(35, 75, 430, 430)
        self.tray_rect = pygame.Rect(35, 520, 430, 100)
        self.cell_size = 430 // 8

    def render(self, game: BlockBlast, gen: int, best_score: int, selected_idx: Optional[int], fps: int, is_paused: bool):
        self.screen.fill(COLOR_BG)

        t1 = self.font_title.render(f"Campeão da Geração {gen}", True, COLOR_CYAN)
        t2 = self.font_title.render(f"Score: {game.score} | Recorde: {best_score}", True, COLOR_GOLD)
        self.screen.blit(t1, (35, 18))
        self.screen.blit(t2, (35, 42))

        pygame.draw.rect(self.screen, COLOR_CARD_BG, self.board_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, self.board_rect, width=1, border_radius=10)

        for r in range(8):
            for c in range(8):
                cx = self.board_rect.left + c * self.cell_size + 3
                cy = self.board_rect.top + r * self.cell_size + 3
                cell_rect = pygame.Rect(cx, cy, self.cell_size - 6, self.cell_size - 6)

                if game.board[r][c] != 0:
                    pygame.draw.rect(self.screen, COLOR_BLUE, cell_rect, border_radius=5)
                    top_light = pygame.Rect(cx + 2, cy + 2, self.cell_size - 10, 4)
                    pygame.draw.rect(self.screen, COLOR_BLUE_LIGHT, top_light, border_radius=2)
                else:
                    pygame.draw.rect(self.screen, COLOR_GRID_EMPTY, cell_rect, border_radius=5)
                    pygame.draw.rect(self.screen, COLOR_GRID_BORDER, cell_rect, width=1, border_radius=5)

        pygame.draw.rect(self.screen, COLOR_CARD_BG, self.tray_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, self.tray_rect, width=1, border_radius=10)

        slot_w = self.tray_rect.width // 3
        mini_size = 14

        for i in range(3):
            sx = self.tray_rect.left + i * slot_w
            sy = self.tray_rect.top
            slot_box = pygame.Rect(sx + 5, sy + 5, slot_w - 10, self.tray_rect.height - 10)

            if i == selected_idx and game.tray[i] is not None:
                pygame.draw.rect(self.screen, (38, 48, 72), slot_box, border_radius=8)
                pygame.draw.rect(self.screen, COLOR_GOLD, slot_box, width=2, border_radius=8)

            piece = game.tray[i]
            if piece is not None:
                pw = piece.width * mini_size
                ph = piece.height * mini_size
                start_x = sx + slot_w // 2 - pw // 2
                start_y = sy + self.tray_rect.height // 2 - ph // 2 - 5

                for dr, dc in piece.blocks:
                    b_rect = pygame.Rect(start_x + dc * mini_size, start_y + dr * mini_size, mini_size - 2, mini_size - 2)
                    pygame.draw.rect(self.screen, piece.color, b_rect, border_radius=2)
            else:
                lbl = self.font_small.render("[Vazio]", True, COLOR_TEXT_MUTED)
                self.screen.blit(lbl, (sx + slot_w // 2 - lbl.get_width() // 2, sy + self.tray_rect.height // 2 - 6))

        status_txt = f"{'[PAUSADO]' if is_paused else f'{fps} FPS'} | [ESPAÇO] Pausa | [R] Reiniciar | [ESC] Sair"
        self.screen.blit(self.font_small.render(status_txt, True, COLOR_TEXT_MUTED), (35, 635))

        pygame.display.flip()


def choose_best_move(game: BlockBlast, net: neat.nn.FeedForwardNetwork) -> Optional[Tuple[int, int, int]]:
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None

    best_val = -float('inf')
    best_move = valid_moves[0]

    for p_idx, r, c in valid_moves:
        features = game.simulate_move_features(p_idx, r, c)
        val = net.activate(features)[0]
        if val > best_val:
            best_val = val
            best_move = (p_idx, r, c)

    return best_move


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Nenhum modelo salvo encontrado em: {MODEL_PATH}")
        print("Execute 'python train.py' primeiro para gerar o melhor campeão.")
        return

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        champion = data["genome"]
        generation = data.get("generation", 1)
        record_score = data.get("score", 0)

        net = neat.nn.FeedForwardNetwork.create(champion, config)
        # Teste de validação
        net.activate([0.0] * config.genome_config.num_inputs)
    except Exception as e:
        print(f"[AVISO] O modelo salvo anterior era de uma versão antiga ({e}).")
        print("Execute 'python train.py' por alguns segundos para salvar o novo modelo inteligente.")
        return

    print("=" * 60)
    print("      REPLAY DO CAMPEÃO SALVO - BLOCK BLAST AI")
    print("=" * 60)
    print(f"Campeão da Geração: {generation} | Recorde: {record_score} pts")
    print("Controles:")
    print("  - [ESPAÇO]: Pausar / Despausar")
    print("  - [SETAS ↑ / ↓]: Ajustar velocidade")
    print("  - [R]: Nova partida")
    print("  - [ESC / Q]: Fechar\n")

    viewer = MatchViewer(width=500, height=670)

    try:
        while True:
            seed = random.randint(1, 999999)
            game = BlockBlast(seed=seed)
            fps = 6
            is_paused = False
            restart = False

            while not game.game_over and not restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            return
                        elif event.key == pygame.K_SPACE:
                            is_paused = not is_paused
                        elif event.key == pygame.K_UP:
                            fps = min(60, fps + 3)
                        elif event.key == pygame.K_DOWN:
                            fps = max(1, fps - 2)
                        elif event.key == pygame.K_r:
                            restart = True

                if is_paused:
                    viewer.render(game, generation, record_score, None, fps, True)
                    viewer.clock.tick(15)
                    continue

                move = choose_best_move(game, net)
                if move is None:
                    break

                p_idx, r, c = move
                viewer.render(game, generation, record_score, p_idx, fps, False)
                viewer.clock.tick(fps)

                game.step(p_idx, r, c)

            if game.game_over and not restart:
                viewer.render(game, generation, record_score, None, fps, False)
                pygame.time.wait(1000)

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
