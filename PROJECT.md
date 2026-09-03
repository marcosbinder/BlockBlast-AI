# Project: Autonomous Block Blast Mobile Gameplay & Touch Calibration Engine

## Architecture
- **Hardware/Device Target**: Motorola Moto G54 5G (`ZF524K4RCM`), 1080x2400 portrait display, 400 dpi.
- **ADB Layer (`adb_client.py`)**: Direct TCP socket transport (`127.0.0.1:5037`) for fast screencap buffer retrieval (~1.3s) and shell input swipe dispatch.
- **Vision Pipeline (`cv_detector.py`)**: Dynamic 4-margin theme calibration (Wood, Blue/Night, Neon, Jungle), SIMD `cv2.absdiff` (<60ms inference), 4-corner bezel background reference + texture variance for 8x8 occupancy, 3-tier shape matching with dynamic binary matrix fallback.
- **Calibration Engine (`empirical_calibrator.py`)**: Empirical live phone probing, ghost shadow detection ($18 \le \Delta E \le 65$), automated anchor offset and vertical lift ($L_y$) learning across all 14 shape families.
- **NEAT Move Planner & FSM Loop (`bot_player.py`, `game.py`)**: Pre-trained NEAT champion (Gen 137, 79 inputs, 4296 pts record), batch 3-piece in-memory permutation search ($3! = 6$ sequences), closed-loop drop verification, adaptive combo particle & refill settling delays, game-over recovery.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | F1: High-Speed Direct ADB Socket Client | Direct TCP connection to ADB server (127.0.0.1:5037) targeting ZF524K4RCM for low-latency screencap and swipe | M1 | ORIGINAL_REQUEST §R4 | DONE |
| 2 | F2: Coordinate Bounds & Safe Clamping | Strict 1080x2400 bounds, safe release clamp ($Y \le 1580$) preventing tray cancel zone release ($Y \ge 1600$) | M1 | ORIGINAL_REQUEST §R4, §R1 | DONE |
| 3 | F3: Dynamic Multi-Theme Adaptation | 4-margin background sampling + Otsu thresholding without hardcoded color constants | M2 | ORIGINAL_REQUEST §R2 | DONE |
| 4 | F4: Robust 8x8 Board Cell Occupancy | Invariant bezel-corner baseline + texture variance ($\sigma > 10$) supporting 0% to 100% board occupancy | M2 | ORIGINAL_REQUEST §R2 | DONE |
| 5 | F5: Tray Piece Segmentation & Fallback | Centroid touch grab extraction $(X_{grab}, Y_{grab})$, slot assignment, and 3-tier dynamic binary shape fallback | M2 | ORIGINAL_REQUEST §R2 | DONE |
| 6 | F6: Empirical Ghost Shadow Calibration | Real-time board ghost shadow detection ($18 \le \Delta E \le 65$) for autonomous anchor/lift self-tuning | M3 | ORIGINAL_REQUEST §R1 | DONE |
| 7 | F7: Batch 3-Piece Permutation Planning | 3! = 6 sequence evaluation in-memory using pre-trained NEAT champion network | M4 | ORIGINAL_REQUEST §R3 | DONE |
| 8 | F8: Live Play FSM & Adaptive Settling | State machine with closed-loop drop verification, combo particle dissipation, and refill bounce settling | M4 | ORIGINAL_REQUEST §R3 | DONE |
| 9 | F9: Game-Over Detection & Clean Recovery | Game-over dialog detection, score extraction, clean restart without hanging or invalid swipes | M4 | ORIGINAL_REQUEST §R3 | DONE |
| 10 | F10: 100% Precision Placement & 10-Round Live Play | Verification of all shape families snapped to intended cells + 10 full rounds (30 pieces) autonomous play | M5 | ORIGINAL_REQUEST §Acceptance Criteria | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | ADB Infrastructure & Safe Coordinate Engine | `adb_client.py`: Direct ADB socket client, raw screencap parsing, coordinate bounds, safe swipe clamping | none | DONE |
| M2 | SIMD Vision & Multi-Theme Detection Pipeline | `cv_detector.py`: Theme-agnostic background calibration, 8x8 occupancy, centroid touch grab, <100ms SIMD pipeline | none | DONE |
| M3 | Empirical Touch Calibration Engine | `empirical_calibrator.py`: Ghost shadow detection, live ADB probing, autonomous calibration matrix generation | M1, M2 | DONE |
| M4 | NEAT 3-Piece Batch Planner & Live Play FSM | `bot_player.py`, `game.py`: Batch 3-piece permutation search, NEAT champion evaluation, robust FSM live loop | M1, M2, M3 | DONE |
| M5 | Final E2E Integration & 10-Round Acceptance | Full system live run on device ZF524K4RCM: 100% placement precision test suite, >=10 rounds continuous play, Tier 5 hardening | M1, M2, M3, M4 | DONE |

## Interface Contracts
### `adb_client.py` ↔ `cv_detector.py` / `bot_player.py`
- `FastADBSocketClient(serial='ZF524K4RCM', host='127.0.0.1', port=5037)`
  - `screencap_cv2() -> np.ndarray (1080x2400 BGR)`
  - `swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool`
  - `tap(x: int, y: int) -> bool`

### `cv_detector.py` ↔ `bot_player.py` / `empirical_calibrator.py`
- `BlockBlastDetector`
  - `detect_state(img: np.ndarray) -> (board_state: np.ndarray [8,8], pieces: List[Dict], meta: Dict)`
  - `detect_ghost_highlights(img: np.ndarray, baseline_img: np.ndarray) -> np.ndarray [8,8] (bool)`
  - `detect_game_over(img: np.ndarray) -> (is_game_over: bool, score: Optional[int])`

### `empirical_calibrator.py` ↔ `bot_player.py`
- `CalibrationProfile`: Loads `calibration_profiles.json` providing `get_finger_target_xy(piece_name, target_row, target_col) -> (int, int)` with safe release clamping $Y \le 1580$.

### `bot_player.py` ↔ `game.py`
- `plan_batch_moves(board_state: np.ndarray, tray_pieces: List[Piece], neat_net) -> List[Tuple[int, int, int]]`: Returns list of `(slot_idx, row, col)` for all pieces in optimal sequence.

## Code Layout
- `adb_client.py`: Fast direct ADB socket communication and screencap stream.
- `cv_detector.py`: SIMD OpenCV theme-agnostic vision detector and board/tray parser.
- `empirical_calibrator.py`: Autonomous empirical touch calibration harness with ghost shadow analysis.
- `game.py`: Core simulation, 42 piece definitions, scoring, combo tracking, 79-feature extraction.
- `bot_player.py`: Live autonomous player loop with NEAT 3-piece batch planner and FSM.
- `calibration_profiles.json`: Calibrated anchor offsets and lift constants for all 14 shape families.
- `tests/`: Comprehensive test suite (Tiers 1-5, 287 passing tests).
