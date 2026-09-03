# E2E Test Infra: Autonomous Block Blast Engine

## Test Philosophy
- Opaque-box, requirement-driven testing based on `ORIGINAL_REQUEST.md`.
- Systematic 4-tier methodology: Category-Partition (Tier 1), Boundary Value Analysis (Tier 2), Pairwise Combinatorial (Tier 3), Real-World Live Workloads (Tier 4), and Adversarial Stress Testing (Tier 5).

## Feature Inventory & Test Coverage Goals
| # | Feature | Requirement | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|-------------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | F1: Fast ADB Direct Socket Client | ORIGINAL_REQUEST §R4 | 5 cases | 5 cases | ✓ | ✓ |
| 2 | F2: Physical Coordinates & Safe Clamping | ORIGINAL_REQUEST §R4, §R1 | 5 cases | 5 cases | ✓ | ✓ |
| 3 | F3: Dynamic Multi-Theme Adaptation | ORIGINAL_REQUEST §R2 | 5 cases | 5 cases | ✓ | ✓ |
| 4 | F4: Invariant 8x8 Board Cell Occupancy | ORIGINAL_REQUEST §R2 | 5 cases | 5 cases | ✓ | ✓ |
| 5 | F5: Tray Piece Segmentation & Fallback | ORIGINAL_REQUEST §R2 | 5 cases | 5 cases | ✓ | ✓ |
| 6 | F6: Empirical Ghost Shadow Calibration | ORIGINAL_REQUEST §R1 | 5 cases | 5 cases | ✓ | ✓ |
| 7 | F7: Batch 3-Piece Permutation Planning | ORIGINAL_REQUEST §R3 | 5 cases | 5 cases | ✓ | ✓ |
| 8 | F8: Live Play FSM & Adaptive Timings | ORIGINAL_REQUEST §R3 | 5 cases | 5 cases | ✓ | ✓ |
| 9 | F9: Game-Over Detection & Score Logging | ORIGINAL_REQUEST §R3 | 5 cases | 5 cases | ✓ | ✓ |
| 10 | F10: 100% Placement Precision & Live Play | ORIGINAL_REQUEST §AC | 5 cases | 5 cases | ✓ | ✓ |

## Test Architecture
- Test runner: `pytest` / python test runner executing `tests/test_e2e_suite.py` and test modules in `tests/`.
- Test harness supports both live ADB execution on Moto G54 5G (`ZF524K4RCM`) and offline deterministic fixture/replay suites.
- Pass/Fail semantics: 100% pass required across all assertions with 0 unhandled exceptions.

## Test Tier Breakdown
- **Tier 1: Feature Coverage (50 cases)**: Isolated unit and functional tests for each of the 10 features under normal/happy-path inputs.
- **Tier 2: Boundary & Corner Cases (50 cases)**: Boundary limits (e.g. Row 0 vs Row 7 releases, Tray cancellation zone $Y \ge 1600$, 0% empty vs 100% full board, 1x1 dot vs 5x5/3x3 huge blocks, high combo multipliers).
- **Tier 3: Cross-Feature Interactions (15 cases)**: Pairwise integration tests (e.g., CV detection under wood theme feeding NEAT batch planning; empirical calibration feeding live ADB swipe execution; combo line clear triggering particle settling delay and subsequent frame capture).
- **Tier 4: Real-World Live Play Scenarios (10 cases)**: End-to-end continuous live play runs on device across $\ge 10$ full rounds (30 pieces), multi-theme board clearing, and game-over recovery.
- **Total Minimum Test Cases**: 125 test cases.
