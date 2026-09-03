"""
train.py - Treinador NEAT v3 com Avaliação PARALELA (multiprocessing).
60 genomas avaliados em 3 workers simultâneos → ~3x mais rápido por geração.
Otimizado para 4GB RAM / 4 cores.
"""

import os
import sys
import time
import csv
import glob
import pickle
import multiprocessing as mp
import neat
import pygame
from typing import List, Tuple, Dict, Any, Optional

from game import BlockBlast
from visualizer import TrainingDashboard
from bot_player import plan_batch_moves

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
HISTORY_FILE = os.path.join(CHECKPOINT_DIR, "training_history.csv")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_champion.pkl")

MAX_MOVES_PER_GAME = 5000  # Joga até morrer naturalmente (limite alto só por segurança extrema)
NUM_GAMES_PER_GENOME = 2
NUM_WORKERS = os.cpu_count() or 4  # Utiliza os 4 núcleos físicos da CPU

# Estado Global
best_historical_score = 0
best_historical_fitness = -float('inf')
last_champion_score = 0
last_gen_genome_scores: Dict[int, int] = {}
current_generation = 0
is_paused = False
turbo_mode = True
last_gen_duration = 0.0
training_start_time = time.time()
gen_durations: List[float] = []
training_history: List[Dict[str, Any]] = []
dashboard_instance: Optional[TrainingDashboard] = None


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "GenScore", "BestRecord", "BestFitness", "Timestamp"])


def load_existing_history():
    global training_history, best_historical_score, best_historical_fitness
    training_history.clear()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gen = int(row.get("Generation", 0))
                    g_score = int(row.get("GenScore", row.get("BestScore", 0)))
                    b_score = int(row.get("BestRecord", row.get("BestScore", g_score)))
                    fit = float(row.get("BestFitness", -999.0))
                    training_history.append({
                        "generation": gen,
                        "gen_score": g_score,
                        "best_score": b_score,
                        "best_fitness": fit
                    })
                    if b_score > best_historical_score:
                        best_historical_score = b_score
                    if fit > best_historical_fitness:
                        best_historical_fitness = fit
        except Exception:
            pass


def get_latest_checkpoint() -> Optional[str]:
    ensure_checkpoint_dir()
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("neat-checkpoint-")]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return os.path.join(CHECKPOINT_DIR, files[-1])


def prune_old_checkpoints(keep_last: int = 5):
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("neat-checkpoint-")]
    if len(files) > keep_last:
        files.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        for old_f in files[:-keep_last]:
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, old_f))
            except OSError:
                pass


def save_champion_model(champion: neat.DefaultGenome, generation: int, score: int, fitness: float):
    ensure_checkpoint_dir()
    data = {
        "genome": champion,
        "generation": generation,
        "score": score,
        "fitness": fitness
    }
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(data, f)


def record_history(generation: int, gen_score: int, record_score: int, fitness: float):
    ensure_checkpoint_dir()
    training_history.append({
        "generation": generation,
        "gen_score": gen_score,
        "best_score": record_score,
        "best_fitness": fitness,
    })
    try:
        with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([generation, gen_score, record_score, f"{fitness:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])
    except Exception as e:
        print(f"[AVISO] Não foi possível gravar CSV de histórico: {e}")


def handle_events():
    global is_paused, turbo_mode, dashboard_instance
    if not pygame.display.get_init():
        return
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            print("\n[SALVO] Treinamento pausado e salvo.")
            sys.exit(0)
        elif event.type == pygame.VIDEORESIZE:
            if dashboard_instance:
                dashboard_instance.update_layout()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if dashboard_instance:
                action = dashboard_instance.handle_click(event.pos)
                if action == "pause":
                    is_paused = not is_paused
                    print(f"[STATUS] Treinamento {'PAUSADO' if is_paused else 'RETOMADO'}.")
                elif action == "turbo":
                    turbo_mode = not turbo_mode
                    print(f"[STATUS] Modo Turbo {'LIGADO' if turbo_mode else 'DESLIGADO'}.")
                elif action == "fullscreen":
                    dashboard_instance.toggle_fullscreen()
                    print(f"[TELA] {'TELA CHEIA' if dashboard_instance.is_fullscreen else 'JANELA'}.")
                elif action == "quit":
                    pygame.quit()
                    print("\n[SALVO] Treinamento salvo e encerrado com sucesso.")
                    sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                print("\n[SALVO] Treinamento pausado e salvo.")
                sys.exit(0)
            elif event.key == pygame.K_SPACE:
                is_paused = not is_paused
                print(f"[STATUS] Treinamento {'PAUSADO' if is_paused else 'RETOMADO'}.")
            elif event.key == pygame.K_t:
                turbo_mode = not turbo_mode
                print(f"[STATUS] Modo Turbo {'LIGADO' if turbo_mode else 'DESLIGADO'}.")
            elif event.key in (pygame.K_f, pygame.K_F11):
                if dashboard_instance:
                    dashboard_instance.toggle_fullscreen()
                    print(f"[TELA] {'TELA CHEIA' if dashboard_instance.is_fullscreen else 'JANELA'}.")


# =============================================
# FUNÇÕES DO WORKER (rodam em processos separados)
# =============================================

def _choose_best_move(game: BlockBlast, net) -> Optional[Tuple[int, int, int]]:
    """Versão standalone para workers."""
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


def _play_one_game(net, seed: int) -> Tuple[float, int]:
    """Joga uma partida completa no worker usando o planejador em lote de 3 peças."""
    game = BlockBlast(seed=seed)
    while not game.game_over and game.moves_count < MAX_MOVES_PER_GAME:
        tray_list = [(p.name, (0, 0)) if p is not None else (None, None) for p in game.tray]
        moves = plan_batch_moves(game.board, tray_list, net)
        if not moves:
            move = _choose_best_move(game, net)
            if move is None:
                break
            game.step(*move)
        else:
            for m in moves:
                game.step(*m)
                if game.game_over:
                    break
    return game.fitness, game.score


_worker_config_cache: Optional[neat.Config] = None


def _get_worker_config(config_path: str) -> neat.Config:
    """Evita reler e reprocessar o arquivo de configuração do disco 60 vezes por geração."""
    global _worker_config_cache
    if _worker_config_cache is None:
        _worker_config_cache = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    return _worker_config_cache


def _eval_single_genome(args) -> Tuple[int, float, int]:
    """Avalia UM genoma com 3 partidas. Roda no worker."""
    genome_id, genome, config_path, seeds = args
    config = _get_worker_config(config_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_fitness = 0.0
    best_score = 0
    for seed in seeds:
        fitness, score = _play_one_game(net, seed)
        total_fitness += fitness
        if score > best_score:
            best_score = score
    avg_fitness = total_fitness / len(seeds)
    return genome_id, avg_fitness, best_score


# =============================================
# AVALIAÇÃO PRINCIPAL (com Pool paralelo)
# =============================================

def eval_genomes(genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config):
    """Avalia a população inteira em paralelo usando multiprocessing."""
    global current_generation, best_historical_score, best_historical_fitness, last_champion_score

    base_seed = (current_generation * 7919 + 42) % (2**31 - 1)
    seeds = [base_seed + i * 104729 for i in range(NUM_GAMES_PER_GENOME)]

    total_genomes = len(genomes)

    # Prepara argumentos para os workers
    worker_args = [
        (genome_id, genome, CONFIG_PATH, seeds)
        for genome_id, genome in genomes
    ]

    # Avaliação paralela com barra de progresso
    gen_best_score = 0
    completed = 0

    genome_map = {gid: g for gid, g in genomes}
    genome_best_scores = {}

    try:
        with mp.Pool(NUM_WORKERS) as pool:
            for genome_id, avg_fitness, best_score in pool.imap_unordered(_eval_single_genome, worker_args):
                completed += 1

                # Atribui fitness diretamente via mapeamento O(1)
                target_genome = genome_map.get(genome_id)
                if target_genome is not None:
                    target_genome.fitness = avg_fitness
                    genome_best_scores[genome_id] = best_score

                if best_score > gen_best_score:
                    gen_best_score = best_score
                if best_score > best_historical_score:
                    best_historical_score = best_score

                # Atualiza dashboard a cada genoma concluído
                handle_events()
                if dashboard_instance:
                    total_elapsed = time.time() - training_start_time
                    avg_duration = (sum(gen_durations) / len(gen_durations)) if gen_durations else last_gen_duration
                    dashboard_instance.update(
                        generation=current_generation,
                        best_score=best_historical_score,
                        last_score=gen_best_score,
                        history=training_history,
                        is_paused=is_paused,
                        turbo=turbo_mode,
                        last_duration=last_gen_duration,
                        avg_duration=avg_duration,
                        total_elapsed=total_elapsed,
                        progress=f"Avaliando {completed}/{total_genomes}"
                    )
    except Exception as e:
        # Fallback sequencial se multiprocessing falhar
        print(f"[AVISO] Paralelo falhou ({e}), usando modo sequencial...")
        for idx, (genome_id, genome) in enumerate(genomes):
            handle_events()
            if dashboard_instance:
                total_elapsed = time.time() - training_start_time
                avg_duration = (sum(gen_durations) / len(gen_durations)) if gen_durations else last_gen_duration
                dashboard_instance.update(
                    generation=current_generation,
                    best_score=best_historical_score,
                    last_score=gen_best_score,
                    history=training_history,
                    is_paused=is_paused,
                    turbo=turbo_mode,
                    last_duration=last_gen_duration,
                    avg_duration=avg_duration,
                    total_elapsed=total_elapsed,
                    progress=f"Avaliando {idx+1}/{total_genomes} (seq)"
                )
            config_local = _get_worker_config(CONFIG_PATH)
            net = neat.nn.FeedForwardNetwork.create(genome, config_local)
            total_fitness = 0.0
            best_score_g = 0
            for seed in seeds:
                fitness, score = _play_one_game(net, seed)
                total_fitness += fitness
                if score > best_score_g:
                    best_score_g = score
            genome.fitness = total_fitness / NUM_GAMES_PER_GENOME
            genome_best_scores[genome_id] = best_score_g
            if best_score_g > gen_best_score:
                gen_best_score = best_score_g
            if best_score_g > best_historical_score:
                best_historical_score = best_score_g

    global last_gen_genome_scores
    last_gen_genome_scores = genome_best_scores
    last_champion_score = gen_best_score


def main():
    global current_generation, last_gen_duration, is_paused, turbo_mode
    global best_historical_score, best_historical_fitness, dashboard_instance

    ensure_checkpoint_dir()
    load_existing_history()

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config NEAT não encontrado: {CONFIG_PATH}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    latest_ckpt = get_latest_checkpoint()
    population = None

    if latest_ckpt:
        try:
            loaded_pop = neat.Checkpointer.restore_checkpoint(latest_ckpt)
            old_inputs = len(loaded_pop.config.genome_config.input_keys)
            new_inputs = config.genome_config.num_inputs
            if old_inputs != new_inputs:
                raise ValueError(f"Entradas: {old_inputs}→{new_inputs}")
            old_pop = len(loaded_pop.population)
            if old_pop != config.pop_size:
                raise ValueError(f"População: {old_pop}→{config.pop_size}")

            population = loaded_pop
            current_generation = population.generation
            print("=" * 65)
            print(f" [AUTO-RESUME] Retomando GERAÇÃO {current_generation + 1} | Recorde: {best_historical_score} pts")
            print(f" [PARALELO] {NUM_WORKERS} workers simultâneos")
            print("=" * 65)
        except Exception as e:
            print("=" * 65)
            print(f" [NOVA CONFIG] {e}")
            print(f" [INICIALIZANDO] IA v3 com {NUM_WORKERS} workers paralelos...")
            print("=" * 65)
            for f in glob.glob(os.path.join(CHECKPOINT_DIR, "neat-checkpoint-*")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.remove(BEST_MODEL_PATH)
            except OSError:
                pass
            best_historical_fitness = -float('inf')
            current_generation = 0
            population = neat.Population(config)
    else:
        print("=" * 65)
        print(f" [NOVO TREINO] IA Estratégica v3 | {NUM_WORKERS} workers paralelos")
        print("=" * 65)
        population = neat.Population(config)

    checkpointer = neat.Checkpointer(
        generation_interval=1,
        time_interval_seconds=None,
        filename_prefix=CHECKPOINT_PREFIX
    )
    population.add_reporter(checkpointer)
    population.add_reporter(neat.StdOutReporter(True))

    dashboard = TrainingDashboard(width=840, height=580)
    dashboard_instance = dashboard

    print("Controles:")
    print("  - Botões na tela (clicáveis com o Mouse)")
    print("  - [ESPAÇO]: Pausar / Despausar")
    print("  - [T]: Alternar Turbo")
    print("  - [F / F11]: Alternar Tela Cheia")
    print("  - [ESC]: Salvar e Sair\n")

    try:
        while True:
            handle_events()

            total_elapsed = time.time() - training_start_time
            avg_duration = (sum(gen_durations) / len(gen_durations)) if gen_durations else last_gen_duration

            if is_paused:
                dashboard.update(
                    generation=current_generation,
                    best_score=best_historical_score,
                    last_score=last_champion_score,
                    history=training_history,
                    is_paused=True,
                    turbo=turbo_mode,
                    last_duration=last_gen_duration,
                    avg_duration=avg_duration,
                    total_elapsed=total_elapsed
                )
                dashboard.clock.tick(15)
                continue

            current_generation = population.generation + 1
            start_t = time.time()

            population.run(eval_genomes, 1)
            current_generation = population.generation
            last_gen_duration = time.time() - start_t
            gen_durations.append(last_gen_duration)
            avg_duration = sum(gen_durations) / len(gen_durations)
            total_elapsed = time.time() - training_start_time

            champion = population.best_genome
            champ_score = last_gen_genome_scores.get(champion.key, last_champion_score)
            print(f">>> [GEN {current_generation}] Score: {champ_score} pts | Recorde: {best_historical_score} pts | {last_gen_duration:.1f}s (Média: {avg_duration:.1f}s)")

            record_history(current_generation, champ_score, best_historical_score, champion.fitness)

            if champion.fitness > best_historical_fitness or champ_score >= best_historical_score:
                if champion.fitness > best_historical_fitness:
                    best_historical_fitness = champion.fitness
                save_champion_model(champion, current_generation, best_historical_score, champion.fitness)

            prune_old_checkpoints(keep_last=5)

            dashboard.update(
                generation=current_generation,
                best_score=best_historical_score,
                last_score=champ_score,
                history=training_history,
                is_paused=False,
                turbo=turbo_mode,
                last_duration=last_gen_duration,
                avg_duration=avg_duration,
                total_elapsed=total_elapsed
            )

            if not turbo_mode:
                dashboard.clock.tick(20)
            else:
                dashboard.clock.tick(120)

    except KeyboardInterrupt:
        print(f"\n[SALVO] Treino interrompido na Geração {current_generation}.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    mp.freeze_support()  # Necessário no Windows para multiprocessing
    main()
