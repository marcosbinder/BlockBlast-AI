"""
visualizer.py - Dashboard Moderno e Leve para Block Blast AI.
Inclui:
- Cards de Métricas (Geração, Recorde, Score Atual, Tempo Total).
- Tempo de Treino, Duração da Última Geração e Média por Geração.
- Botões Interativos Clicáveis (Pausar/Retomar, Turbo/Normal, Tela Cheia, Salvar e Sair).
- Efeito Hover com feedback visual dinâmico.
- Totalmente leve e compatível com 4GB RAM e telas redimensionáveis.
"""

import time
import pygame
from typing import List, Dict, Any, Optional, Tuple

# Paleta de Cores Moderna e Elegante (Dark Theme)
COLOR_BG = (10, 12, 18)
COLOR_CARD_BG = (18, 22, 32)
COLOR_CARD_BORDER = (35, 42, 60)
COLOR_CARD_HOVER = (28, 34, 48)

COLOR_TEXT_MAIN = (245, 248, 255)
COLOR_TEXT_MUTED = (130, 142, 168)
COLOR_TEXT_DIM = (85, 95, 120)

COLOR_CYAN = (0, 230, 200)
COLOR_GOLD = (255, 205, 50)
COLOR_GREEN = (45, 215, 120)
COLOR_BLUE = (65, 150, 255)
COLOR_PURPLE = (175, 100, 245)
COLOR_RED = (245, 75, 90)
COLOR_ORANGE = (255, 155, 50)
COLOR_GRID_LINE = (24, 28, 42)


def format_duration(seconds: float) -> str:
    """Formata segundos em HH:MM:SS ou MM:SS."""
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


class TrainingDashboard:
    """Dashboard com gráficos em tempo real, métricas de tempo e botões clicáveis."""
    def __init__(self, width: int = 840, height: int = 580):
        pygame.init()
        pygame.display.set_caption("Block Blast AI - Monitor de Treinamento")
        self.win_width = width
        self.win_height = height
        self.width = width
        self.height = height
        self.is_fullscreen = False

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # Áreas dos Botões Clicáveis
        self.btn_pause_rect = pygame.Rect(0, 0, 10, 10)
        self.btn_turbo_rect = pygame.Rect(0, 0, 10, 10)
        self.btn_fullscreen_rect = pygame.Rect(0, 0, 10, 10)
        self.btn_quit_rect = pygame.Rect(0, 0, 10, 10)

        self.update_layout()

    def update_layout(self):
        """Recalcula dimensões de acordo com o tamanho da tela."""
        w, h = self.screen.get_size()
        self.width, self.height = w, h

        # Escala responsiva suave
        scale = max(1.0, min(2.0, w / 840.0))
        self.scale = scale

        title_size = int(14 * scale)
        big_size = int(22 * scale)
        lbl_size = int(11 * scale)
        main_size = int(12 * scale)
        small_size = int(10 * scale)

        sys_font = "Segoe UI" if pygame.font.match_font("Segoe UI") else None
        self.font_title = pygame.font.SysFont(sys_font, title_size, bold=True) if sys_font else pygame.font.Font(None, int(22 * scale))
        self.font_big_num = pygame.font.SysFont(sys_font, big_size, bold=True) if sys_font else pygame.font.Font(None, int(30 * scale))
        self.font_card_lbl = pygame.font.SysFont(sys_font, lbl_size, bold=True) if sys_font else pygame.font.Font(None, int(15 * scale))
        self.font_main = pygame.font.SysFont(sys_font, main_size) if sys_font else pygame.font.Font(None, int(17 * scale))
        self.font_btn = pygame.font.SysFont(sys_font, main_size, bold=True) if sys_font else pygame.font.Font(None, int(17 * scale))
        self.font_small = pygame.font.SysFont(sys_font, small_size) if sys_font else pygame.font.Font(None, int(14 * scale))

        margin = int(16 * scale)
        card_h = int(74 * scale)
        gap = int(12 * scale)

        # 4 Cards Superiores (Geração, Recorde, Score Atual, Tempo Total)
        card_w = (w - (margin * 2) - (gap * 3)) // 4
        self.cards_rect = [
            pygame.Rect(margin + i * (card_w + gap), margin, card_w, card_h)
            for i in range(4)
        ]

        # Painel Inferior (Progresso + Botões Clicáveis)
        bottom_h = int(96 * scale)
        self.bottom_rect = pygame.Rect(margin, h - bottom_h - margin, w - (margin * 2), bottom_h)

        # Gráfico Central
        chart_top = margin + card_h + gap
        chart_h = self.bottom_rect.top - chart_top - gap
        self.chart_rect = pygame.Rect(margin, chart_top, w - (margin * 2), max(140, chart_h))

        # Posicionamento dos 4 Botões Clicáveis na Barra Inferior
        btn_y = self.bottom_rect.top + int(48 * scale)
        btn_h = int(36 * scale)
        btn_gap = int(10 * scale)
        btn_w = (self.bottom_rect.width - int(32 * scale) - (btn_gap * 3)) // 4
        btn_start_x = self.bottom_rect.left + int(16 * scale)

        self.btn_pause_rect = pygame.Rect(btn_start_x, btn_y, btn_w, btn_h)
        self.btn_turbo_rect = pygame.Rect(btn_start_x + (btn_w + btn_gap), btn_y, btn_w, btn_h)
        self.btn_fullscreen_rect = pygame.Rect(btn_start_x + (btn_w + btn_gap) * 2, btn_y, btn_w, btn_h)
        self.btn_quit_rect = pygame.Rect(btn_start_x + (btn_w + btn_gap) * 3, btn_y, btn_w, btn_h)

    def toggle_fullscreen(self):
        """Alterna entre Tela Cheia e Modo Janela."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.win_width, self.win_height), pygame.RESIZABLE)
        self.update_layout()

    def handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        """Detecta clique do mouse nos botões interativos."""
        if self.btn_pause_rect.collidepoint(pos):
            return "pause"
        elif self.btn_turbo_rect.collidepoint(pos):
            return "turbo"
        elif self.btn_fullscreen_rect.collidepoint(pos):
            return "fullscreen"
        elif self.btn_quit_rect.collidepoint(pos):
            return "quit"
        return None

    def draw_cards(self, generation: int, best_score: int, last_score: int, total_elapsed: float):
        tempo_str = format_duration(total_elapsed)
        cards_data = [
            ("GERAÇÃO ATUAL", str(generation), COLOR_CYAN, "População: 60"),
            ("RECORDE HISTÓRICO", f"{best_score} pts", COLOR_GOLD, "Melhor Campeão"),
            ("SCORE DA GERAÇÃO", f"{last_score} pts", COLOR_GREEN, "Ao vivo"),
            ("TEMPO TOTAL", tempo_str, COLOR_BLUE, "Tempo de treino"),
        ]

        for i, (label, val, col, sub) in enumerate(cards_data):
            rect = self.cards_rect[i]
            pygame.draw.rect(self.screen, COLOR_CARD_BG, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_CARD_BORDER, rect, width=1, border_radius=8)

            lbl_surf = self.font_card_lbl.render(label, True, COLOR_TEXT_MUTED)
            val_surf = self.font_big_num.render(val, True, col)
            sub_surf = self.font_small.render(sub, True, COLOR_TEXT_DIM)

            self.screen.blit(lbl_surf, (rect.left + 12, rect.top + 8))
            self.screen.blit(val_surf, (rect.left + 12, rect.top + 28))
            self.screen.blit(sub_surf, (rect.left + 12, rect.top + 54))

    def draw_chart(self, history: List[Dict[str, Any]], best_score: int,
                   last_duration: float, avg_duration: float):
        pygame.draw.rect(self.screen, COLOR_CARD_BG, self.chart_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, self.chart_rect, width=1, border_radius=10)

        # Título
        title = self.font_title.render("EVOLUÇÃO DAS PONTUAÇÕES POR GERAÇÃO", True, COLOR_TEXT_MAIN)
        self.screen.blit(title, (self.chart_rect.left + 16, self.chart_rect.top + 12))

        # Badges de Tempo no Topo do Gráfico
        badge_x = self.chart_rect.left + 16 + title.get_width() + 18
        if last_duration > 0 or avg_duration > 0:
            info_txt = f"Última: {last_duration:.1f}s  |  Média: {avg_duration:.1f}s/gen"
            info_surf = self.font_small.render(info_txt, True, COLOR_TEXT_MUTED)
            self.screen.blit(info_surf, (badge_x, self.chart_rect.top + 14))

        # Legenda
        leg_cur = self.font_small.render("● Geração Atual (Ciano)", True, COLOR_CYAN)
        leg_rec = self.font_small.render("● Recorde Histórico (Dourado)", True, COLOR_GOLD)
        self.screen.blit(leg_cur, (self.chart_rect.right - 340, self.chart_rect.top + 14))
        self.screen.blit(leg_rec, (self.chart_rect.right - 175, self.chart_rect.top + 14))

        if not history or len(history) < 2:
            msg = self.font_main.render("Coletando primeiras gerações para gerar o gráfico...", True, COLOR_TEXT_MUTED)
            self.screen.blit(msg, (self.chart_rect.left + 20, self.chart_rect.top + 60))
            return

        recent_history = history[-80:] if len(history) > 80 else history
        gen_scores = [h.get("gen_score", h.get("best_score", 0)) for h in recent_history]
        record_scores = [h.get("best_score", 0) for h in recent_history]

        max_val = max(10, max(max(gen_scores), max(record_scores)))

        plot_left = self.chart_rect.left + 45
        plot_right = self.chart_rect.right - 25
        plot_top = self.chart_rect.top + 45
        plot_bottom = self.chart_rect.bottom - 22
        plot_w = plot_right - plot_left
        plot_h = max(30, plot_bottom - plot_top)

        # Linhas de grade e valores
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = plot_bottom - int(frac * plot_h)
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (plot_left, y), (plot_right, y), 1)
            val_txt = self.font_small.render(f"{int(frac * max_val)}", True, COLOR_TEXT_DIM)
            self.screen.blit(val_txt, (plot_left - 38, y - 6))

        step_x = plot_w / (len(recent_history) - 1)

        # Curva Dourada (Recorde Histórico)
        rec_points = []
        for i, s in enumerate(record_scores):
            x = plot_left + int(i * step_x)
            y = plot_bottom - int((s / max_val) * plot_h)
            rec_points.append((x, y))

        if len(rec_points) >= 2:
            pygame.draw.lines(self.screen, COLOR_GOLD, False, rec_points, 3)

        # Curva Ciano (Score da Geração)
        cur_points = []
        for i, s in enumerate(gen_scores):
            x = plot_left + int(i * step_x)
            y = plot_bottom - int((s / max_val) * plot_h)
            cur_points.append((x, y))

        if len(cur_points) >= 2:
            pygame.draw.lines(self.screen, COLOR_CYAN, False, cur_points, 2)
            for pt in cur_points[-8:]:
                pygame.draw.circle(self.screen, COLOR_CYAN, pt, 4)
                pygame.draw.circle(self.screen, COLOR_BG, pt, 2)

    def draw_interactive_button(self, rect: pygame.Rect, text: str,
                                color: Tuple[int, int, int], mouse_pos: Tuple[int, int]):
        """Desenha um botão clicável com efeito hover estilizado."""
        is_hover = rect.collidepoint(mouse_pos)

        # Cores com hover
        bg_col = (min(255, color[0] // 5 + 18), min(255, color[1] // 5 + 22), min(255, color[2] // 5 + 32))
        border_col = color if is_hover else (color[0] // 2, color[1] // 2, color[2] // 2)

        if is_hover:
            bg_col = (min(255, bg_col[0] + 15), min(255, bg_col[1] + 15), min(255, bg_col[2] + 20))

        pygame.draw.rect(self.screen, bg_col, rect, border_radius=6)
        pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=6)

        txt_surf = self.font_btn.render(text, True, color if is_hover else COLOR_TEXT_MAIN)
        txt_x = rect.left + (rect.width - txt_surf.get_width()) // 2
        txt_y = rect.top + (rect.height - txt_surf.get_height()) // 2
        self.screen.blit(txt_surf, (txt_x, txt_y))

    def draw_bottom_bar(self, is_paused: bool, turbo: bool, last_duration: float,
                        avg_duration: float, progress: str = ""):
        pygame.draw.rect(self.screen, COLOR_CARD_BG, self.bottom_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, self.bottom_rect, width=1, border_radius=10)

        mouse_pos = pygame.mouse.get_pos()

        # Status textual e tempos detalhados
        if is_paused:
            status_txt = "⏸ PAUSADO"
            status_col = COLOR_PURPLE
        elif progress:
            status_txt = f"⚡ {progress}"
            status_col = COLOR_CYAN
        elif turbo:
            status_txt = "⚡ Modo Turbo Ativo"
            status_col = COLOR_CYAN
        else:
            status_txt = "Modo Normal"
            status_col = COLOR_GREEN

        t_stat = self.font_main.render(status_txt, True, status_col)
        self.screen.blit(t_stat, (self.bottom_rect.left + 16, self.bottom_rect.top + 8))

        # Indicador de tempo no canto direito da barra de status
        metrics_str = f"Última Gen: {last_duration:.1f}s   |   Média: {avg_duration:.1f}s/gen"
        t_metrics = self.font_small.render(metrics_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(t_metrics, (self.bottom_rect.right - t_metrics.get_width() - 16, self.bottom_rect.top + 10))

        # Barra de Progresso Visual Suave
        bar_x = self.bottom_rect.left + 16
        bar_y = self.bottom_rect.top + 30
        bar_w = self.bottom_rect.width - 32
        bar_h = 7
        pygame.draw.rect(self.screen, (26, 30, 44), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

        if progress:
            try:
                parts = progress.split("/")
                if len(parts) == 2:
                    curr_val = int(''.join(filter(str.isdigit, parts[0])))
                    total_val = int(''.join(filter(str.isdigit, parts[1])))
                    if total_val > 0:
                        fill_w = int(bar_w * (curr_val / total_val))
                        if fill_w > 0:
                            pygame.draw.rect(self.screen, COLOR_CYAN, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
            except (ValueError, IndexError):
                pass

        # 4 Botões Clicáveis Interativos
        pause_txt = "▶ RETOMAR" if is_paused else "⏸ PAUSAR"
        pause_col = COLOR_GREEN if is_paused else COLOR_PURPLE
        self.draw_interactive_button(self.btn_pause_rect, pause_txt, pause_col, mouse_pos)

        turbo_txt = "⚡ TURBO: LIGADO" if turbo else "🐢 MODO NORMAL"
        turbo_col = COLOR_CYAN if turbo else COLOR_ORANGE
        self.draw_interactive_button(self.btn_turbo_rect, turbo_txt, turbo_col, mouse_pos)

        screen_txt = "🗗 JANELA" if self.is_fullscreen else "⛶ TELA CHEIA"
        self.draw_interactive_button(self.btn_fullscreen_rect, screen_txt, COLOR_BLUE, mouse_pos)

        self.draw_interactive_button(self.btn_quit_rect, "💾 SALVAR E SAIR", COLOR_RED, mouse_pos)

    def update(self, generation: int, best_score: int, last_score: int, 
               history: List[Dict[str, Any]], is_paused: bool, turbo: bool,
               last_duration: float, avg_duration: float = 0.0,
               total_elapsed: float = 0.0, progress: str = ""):
        self.screen.fill(COLOR_BG)
        self.draw_cards(generation, best_score, last_score, total_elapsed)
        self.draw_chart(history, best_score, last_duration, avg_duration)
        self.draw_bottom_bar(is_paused, turbo, last_duration, avg_duration, progress)
        pygame.display.flip()
