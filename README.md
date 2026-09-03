# 🎮 Block Blast AI - Autonomous Mobile Bot & NEAT Evolution

Um sistema autônomo completo de Inteligência Artificial para jogar **Block Blast** diretamente no celular Android (via ADB / scrcpy) e treinar redes neurais com **NEAT (NeuroEvolution of Augmenting Topologies)**.

🏆 **Recorde Atual do Campeão:** **15.433 pontos** (Fitness: **74.887**)

---

## 🌟 Principais Funcionalidades

1. **Bot Autônomo Mobile em Tempo Real (ot_player.py)**:
   - Conexão direta com dispositivo Android via ADB / scrcpy.
   - Máquina de Estados Finitos (FSM) com ciclo autônomo contínuo: captura, detecção, planejamento em lote, execução por gestos e verificação de encaixe.
   - **Compensação Física de Toque**: Modelo de compensação do multiplicador de arraste da engine Unity (DRAG_GAIN = 1.4x) com offset vertical dinâmico (275px), garantindo drops milimetricamente precisos no grid 8x8.
   - Safe Release Clamping (impede cancelamento acidental na área da bandeja).

2. **Visão Computacional OpenCV Multi-Tema (cv_detector.py)**:
   - Detecção em tempo real de matriz binária de ocupação 8x8 e classificação das 42 formas de blocos.
   - Adaptação dinâmica a qualquer tema (azul clássico, vermelho, ciano, escuro) com limiarização robusta contra ruídos.

3. **Treinador NEAT com Lookahead em Lote (	rain.py)**:
   - Avaliação com **Batch 3-Piece Lookahead**: em vez de jogadas míopes de 1 peça, avalia as 6 permutações das 3 peças da bandeja juntas com busca em feixe (*beam search*).
   - Multiprocessing paralelo utilizando todos os núcleos da CPU (4 workers).
   - Dashboard gráfico interativo em Pygame com gráficos de evolução, métricas em tempo real e modo Turbo.

---

## 📁 Estrutura do Projeto

`
├── bot_player.py              # Bot FSM autônomo para jogo ao vivo no celular
├── cv_detector.py             # Pipeline de visão computacional OpenCV SIMD
├── adb_client.py              # Cliente ADB de baixa latência e injeção de gestos
├── game.py                    # Motor oficial do Block Blast com as 42 peças oficiais
├── train.py                   # Treinador paralelo NEAT com visualizador Pygame
├── visualizer.py              # Interface gráfica do dashboard de treinamento
├── config-feedforward         # Configuração dos hiperparâmetros genéticos do NEAT
├── calibration_profiles.json  # Perfis e âncoras de calibração empírica das peças
├── checkpoints/
│   ├── best_champion.pkl     # Rede neural campeã atual (15.433 pts)
│   └── training_history.csv   # Histórico completo de evolução por geração
└── requirements.txt           # Dependências do projeto
`

---

## 🚀 Instalação e Requisitos

### Pré-requisitos
- Python 3.10+ (testado em Python 3.14)
- Android com **Depuração USB** ativada (e scrcpy/adb configurado)

### Instalar dependências
`ash
pip install -r requirements.txt
`

---

## 🕹️ Como Usar

### 1. Executar o Treinamento NEAT
`ash
python train.py
`
**Controles no Treinador:**
* **[ESPAÇO]**: Pausar / Retomar o treino.
* **[T]**: Alternar modo Turbo.
* **[F] / [F11]**: Alternar Tela Cheia.
* **[ESC]**: Salvar checkpoint e sair.

### 2. Executar o Bot Autônomo no Celular
`ash
# Testar apenas o plano de jogadas da tela atual
python bot_player.py --test-plan

# Executar 1 jogada de teste
python bot_player.py --test-one-move

# Iniciar o loop autônomo contínuo (FSM)
python bot_player.py
`

### 3. Assistir ao Replay do Campeão
`ash
python replay.py
`