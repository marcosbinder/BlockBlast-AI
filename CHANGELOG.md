# Changelog

All notable changes to the `BlockBlast-AI` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-15

### Core Engine & Architecture
- **Single Source of Truth Planner (`planner.py`)**:
  - Unified feature extraction (79 normalized inputs) and 3-piece lookahead permutation search ($3! = 6$ tray piece sequences).
  - Implemented sequential line/column clear with immediate matrix updates and Combo Streak propagation across sub-steps.
  - Centralized state evaluation shared seamlessly across training (`train.py`), live autonomous play (`bot_player.py`), and replay verification (`replay.py`).
- **NEAT Neuroevolution Model**:
  - Pre-trained champion neural network (817+ generations, 15,433+ points high-score record).
  - Tactical heuristic evaluation including anti-hole isolation penalty, 3x3 center core preservation, perimeter placement bias, and row/column line-transition minimizer.

### Computer Vision & Detection (`cv_detector.py`)
- **Theme-Agnostic 8x8 Grid Classifier**:
  - Introduced invariant 4-corner bezel background sampling (`c_tl`, `c_tr`, `c_bl`, `c_br`) replacing vulnerable darkest-cell estimation.
  - Verified 100% accurate classification across 0% empty, crowded, and 100% full boards in Wood, Blue/Night, Neon, and Jungle themes.
- **Noise & Animation Guarding**:
  - Added strict aspect ratio and minimum contour area filtering in `_classify_shape` to prevent explosion particles and combo pop-ups from producing false `1x1` dot detections.
  - Dynamic fallback shape matrix generator for unexpected or seasonal pieces.

### Touch Physics & ADB Client (`adb_client.py` & `empirical_calibrator.py`)
- **Safe Release Clamping**:
  - Corrected `MAX_SAFE_RELEASE_Y` constant to strictly enforce $Y \le 1580\text{ px}$, preventing accidental drops into the tray cancellation hazard zone ($Y \ge 1600\text{ px}$).
- **Kinematic Drag Model**:
  - Precise touch coordinate synthesis accounting for dynamic drag gain (`DRAG_GAIN = 1.4`), vertical finger lift ($275\text{ px}$ offset), and geometric bounding-box center of mass per piece geometry.
- **Robust Socket Transport**:
  - Direct TCP ADB connection (`127.0.0.1:5037`) with length framing, chunked screencap reading, timeouts, and safe recovery.

### Testing & Infrastructure
- **Comprehensive 5-Tier Test Suite**:
  - 287 automated tests covering unit features (Tier 1), boundary/corner cases (Tier 2), cross-module pairwise interactions (Tier 3), real-world multi-round workloads (Tier 4), and adversarial/deadlock stress tests (Tier 5).
  - 100% pass rate with zero warnings.
- **Modern Packaging & CI/CD**:
  - Migrated to PEP 621 compliant `pyproject.toml`.
  - Configured multi-OS GitHub Actions workflow (`ubuntu-latest` and `windows-latest`) running under headless Pygame (`SDL_VIDEODRIVER=dummy`).
  - Automated GitHub Releases workflow on version tags with packaged artifacts.
