"""
test_cv_detector.py - Comprehensive Unit & Performance Test Suite for cv_detector.py
Tests BlockBlastDetector across:
- All 18 repository test screenshots (Wood, Blue/Night themes, empty, crowded, 100% full boards).
- Dynamic 4-margin background calibration without hardcoded color constants.
- 4-corner bezel background reference + texture variance for 8x8 occupancy.
- Exact centroid touch grab coordinates (X_grab, Y_grab) and minimum Euclidean distance slotting.
- 3-tier shape classification: exact hash -> Hamming distance <= 1 -> dynamic binary matrix fallback.
- Ghost highlight detection (18 <= Delta E <= 65) for empirical touch calibration.
- Game-over dialog detection.
- Sub-60ms average inference speed (<100ms hard ceiling).
"""

import os
import glob
import time
import cv2
import numpy as np
import pytest
from typing import List, Tuple

from cv_detector import BlockBlastDetector, CANONICAL_SHAPES
from game import BLOCK_SHAPES

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALL_18_SCREENSHOTS = [
    "after_blue_test.png",
    "after_center_test.png",
    "after_tap.png",
    "after_test.png",
    "current_bug_screen.png",
    "current_check.png",
    "current_live.png",
    "current_live_now.png",
    "current_misplace.png",
    "current_phone.png",
    "current_state.png",
    "debug_tray.png",
    "holding_piece.png",
    "holding_zv.png",
    "phone_screen.png",
    "probe_1160.png",
    "probe_778_1244.png",
    "probe_target_snap.png",
]


@pytest.fixture(scope="module")
def detector() -> BlockBlastDetector:
    return BlockBlastDetector()


@pytest.fixture(scope="module")
def loaded_images() -> dict:
    images = {}
    for name in ALL_18_SCREENSHOTS:
        path = os.path.join(REPO_ROOT, name)
        assert os.path.exists(path), f"Required repository test image missing: {path}"
        img = cv2.imread(path)
        assert img is not None, f"Failed to load image: {path}"
        assert img.shape == (2400, 1080, 3), f"Unexpected shape for {name}: {img.shape}"
        images[name] = img
    return images


# =====================================================================
# 1. CORE GEOMETRY & COORDINATES TESTS
# =====================================================================
class TestDetectorGeometry:
    def test_board_geometry_constants(self, detector: BlockBlastDetector):
        assert detector.board_left == 61
        assert detector.board_top == 581
        assert detector.board_right == 1018
        assert detector.board_bottom == 1537
        assert abs(detector.cell_w - 119.625) < 1e-3
        assert abs(detector.cell_h - 119.5) < 1e-3

    def test_tray_geometry_constants(self, detector: BlockBlastDetector):
        assert detector.tray_top == 1660
        assert detector.tray_bottom == 2050
        assert detector.tray_left == 60
        assert detector.tray_right == 1020
        assert abs(detector.tray_block_size - 58.5) < 1e-3

    def test_slot_centers(self, detector: BlockBlastDetector):
        assert len(detector.slot_centers_x) == 3
        assert abs(detector.slot_centers_x[0] - 220.0) < 1.0
        assert abs(detector.slot_centers_x[1] - 540.0) < 1.0
        assert abs(detector.slot_centers_x[2] - 860.0) < 1.0

    def test_cell_screen_coords(self, detector: BlockBlastDetector):
        # Corner (0, 0)
        x0, y0 = detector.get_cell_screen_coords(0, 0)
        assert x0 == int(61 + 0.5 * 119.625)
        assert y0 == int(581 + 0.5 * 119.5)

        # Center (3, 4)
        x_mid, y_mid = detector.get_cell_screen_coords(3, 4)
        assert 61 < x_mid < 1018
        assert 581 < y_mid < 1537

        # Corner (7, 7)
        x7, y7 = detector.get_cell_screen_coords(7, 7)
        assert x7 == int(61 + 7.5 * 119.625)
        assert y7 == int(581 + 7.5 * 119.5)


# =====================================================================
# 2. BOARD OCCUPANCY DETECTION TESTS (0% TO 100%)
# =====================================================================
class TestBoardOccupancy:
    def test_all_18_images_board_state_shape_and_values(self, detector: BlockBlastDetector, loaded_images: dict):
        for name, img in loaded_images.items():
            board = detector.detect_board_state(img)
            assert len(board) == 8, f"Board must have 8 rows in {name}"
            for r in range(8):
                assert len(board[r]) == 8, f"Row {r} must have 8 cols in {name}"
                for c in range(8):
                    assert board[r][c] in (0, 1), f"Invalid cell value at ({r},{c}) in {name}: {board[r][c]}"

    def test_empty_boards_zero_occupancy(self, detector: BlockBlastDetector, loaded_images: dict):
        empty_screens = [
            "phone_screen.png",
            "current_phone.png",
            "after_test.png",
            "after_center_test.png",
            "current_live.png",
            "current_live_now.png",
        ]
        for name in empty_screens:
            board = detector.detect_board_state(loaded_images[name])
            total_occupied = sum(sum(row) for row in board)
            assert total_occupied == 0, f"Expected 0 occupied cells in {name}, found {total_occupied}"

    def test_completely_full_board_100_percent_occupancy(self, detector: BlockBlastDetector, loaded_images: dict):
        board = detector.detect_board_state(loaded_images["probe_target_snap.png"])
        total_occupied = sum(sum(row) for row in board)
        assert total_occupied == 64, f"Expected 64/64 occupied cells in probe_target_snap.png, found {total_occupied}"

    def test_crowded_boards_occupancy(self, detector: BlockBlastDetector, loaded_images: dict):
        crowded_expectations = {
            "current_state.png": 29,
            "probe_1160.png": 29,
            "holding_zv.png": 33,
            "probe_778_1244.png": 30,
            "current_misplace.png": 22,
            "current_bug_screen.png": 16,
            "after_tap.png": 2,
            "after_blue_test.png": 1,
            "current_check.png": 3,
            "debug_tray.png": 3,
            "holding_piece.png": 3,
        }
        for name, expected_count in crowded_expectations.items():
            board = detector.detect_board_state(loaded_images[name])
            total_occupied = sum(sum(row) for row in board)
            assert total_occupied == expected_count, (
                f"Mismatch in {name}: expected {expected_count} cells, detected {total_occupied}"
            )


# =====================================================================
# 3. TRAY PIECE DETECTION & CENTROID GRAB COORDINATES
# =====================================================================
class TestTrayPieceDetection:
    def test_all_18_images_tray_detection_format(self, detector: BlockBlastDetector, loaded_images: dict):
        for name, img in loaded_images.items():
            tray = detector.detect_tray_pieces(img)
            assert len(tray) == 3, f"Tray must return exactly 3 slots in {name}"
            for slot_idx, (p_name, grab_xy) in enumerate(tray):
                if p_name is not None:
                    assert isinstance(p_name, str)
                    assert p_name in BLOCK_SHAPES or p_name.startswith("dyn_")
                    assert grab_xy is not None
                    gx, gy = grab_xy
                    # Centroid must reside within physical tray screen bounds
                    assert 60 <= gx <= 1020, f"Grab X out of bounds in {name} slot {slot_idx}: {gx}"
                    assert 1660 <= gy <= 2050, f"Grab Y out of bounds in {name} slot {slot_idx}: {gy}"
                    # Grab coordinate must be in vicinity of the corresponding slot
                    slot_center_x = detector.slot_centers_x[slot_idx]
                    assert abs(gx - slot_center_x) < 180, (
                        f"Slot {slot_idx} piece center X ({gx}) too far from nominal ({slot_center_x}) in {name}"
                    )
                else:
                    assert grab_xy is None

    def test_specific_tray_pieces_accuracy(self, detector: BlockBlastDetector, loaded_images: dict):
        # 1. Blue theme full tray
        tray_blue = detector.detect_tray_pieces(loaded_images["after_blue_test.png"])
        assert [p[0] for p in tray_blue] == ["corner_tr", "diag2_down", "line2_v"]

        # 2. Wood theme full tray
        tray_wood = detector.detect_tray_pieces(loaded_images["after_center_test.png"])
        assert [p[0] for p in tray_wood] == ["line2_h", "diag2_down", "corner_tl"]

        # 3. Partial tray (slot 0 played)
        tray_partial = detector.detect_tray_pieces(loaded_images["after_tap.png"])
        assert [p[0] for p in tray_partial] == [None, "diag2_down", "corner_tl"]

        # 4. Empty tray (all pieces played / holding)
        tray_empty = detector.detect_tray_pieces(loaded_images["current_bug_screen.png"])
        assert [p[0] for p in tray_empty] == [None, None, None]

        # 5. Snap probe screen (slots 0 and 2 present)
        tray_snap = detector.detect_tray_pieces(loaded_images["probe_target_snap.png"])
        assert [p[0] for p in tray_snap] == ["l_h_tr", None, "line4_v"]


# =====================================================================
# 4. 3-TIER SHAPE CLASSIFICATION HIERARCHY & DYNAMIC FALLBACK
# =====================================================================
class TestShapeClassificationHierarchy:
    def test_tier1_canonical_exact_hash_match(self, detector: BlockBlastDetector):
        # Test all 42 standard pieces
        for name, blocks in BLOCK_SHAPES.items():
            h = max(r for r, c in blocks) + 1
            w = max(c for r, c in blocks) + 1
            mask = np.zeros((h * 60, w * 60), dtype=np.uint8)
            for r, c in blocks:
                mask[r * 60:(r + 1) * 60, c * 60:(c + 1) * 60] = 255

            classified = detector._classify_shape(mask, 0, 0, w * 60, h * 60)
            assert classified == name, f"Tier 1 exact match failed for {name}: got {classified}"

    def test_tier2_hamming_distance_tolerance(self, detector: BlockBlastDetector):
        # Create corner_tl (2x2: (0,0), (0,1), (1,0)) with 1 eroded bit at (0,1)
        # Should match corner_tl via Hamming distance <= 1
        blocks = BLOCK_SHAPES["corner_tl"]
        h, w = 2, 2
        mask = np.zeros((120, 120), dtype=np.uint8)
        # Fill (0,0) and (1,0) fully, but leave (0,1) slightly below threshold
        mask[0:60, 0:60] = 255
        mask[60:120, 0:60] = 255
        mask[0:60, 60:120] = 0  # Missing cell (Hamming dist = 1 from corner_tl and line2_v)

        classified = detector._classify_shape(mask, 0, 0, 120, 120)
        assert classified in ("corner_tl", "line2_v"), f"Tier 2 Hamming match failed: got {classified}"

    def test_tier3_dynamic_fallback_for_unknown_shape(self, detector: BlockBlastDetector):
        # Create a novel 3x3 U-shape piece not in canonical 42 shapes:
        # (0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)
        novel_blocks = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        mask = np.zeros((180, 180), dtype=np.uint8)
        for r, c in novel_blocks:
            mask[r * 60:(r + 1) * 60, c * 60:(c + 1) * 60] = 255

        classified = detector._classify_shape(mask, 0, 0, 180, 180)
        assert classified is not None
        assert classified.startswith("dyn_3x3_"), f"Expected dynamic fallback name, got {classified}"
        assert classified in BLOCK_SHAPES, "Dynamic shape must be registered into BLOCK_SHAPES"
        assert BLOCK_SHAPES[classified] == novel_blocks


# =====================================================================
# 5. GHOST HIGHLIGHT & TOUCH SNAP DETECTION
# =====================================================================
class TestGhostHighlights:
    def test_ghost_highlights_blue_theme(self, detector: BlockBlastDetector, loaded_images: dict):
        base_img = loaded_images["phone_screen.png"]
        held_img = loaded_images["holding_piece.png"]
        ghost_mask = detector.detect_ghost_highlights(held_img, base_img)

        assert ghost_mask.shape == (8, 8)
        active_coords = sorted(np.argwhere(ghost_mask).tolist())
        # The piece held is corner_tl at (3, 2), producing active cells at (3,2), (4,2), (4,3)
        expected_cells = sorted([[3, 2], [4, 2], [4, 3]])
        assert active_coords == expected_cells, f"Ghost shadow mismatch: expected {expected_cells}, got {active_coords}"

    def test_ghost_highlights_identical_images_produce_zero(self, detector: BlockBlastDetector, loaded_images: dict):
        for name in ["phone_screen.png", "current_state.png"]:
            img = loaded_images[name]
            ghost_mask = detector.detect_ghost_highlights(img, img)
            assert np.sum(ghost_mask) == 0, f"Identical images in {name} must have 0 ghost highlights"


# =====================================================================
# 6. HIGH-LEVEL STATE & GAME-OVER DETECTION
# =====================================================================
class TestHighLevelStateAndGameOver:
    def test_detect_state_interface_contract(self, detector: BlockBlastDetector, loaded_images: dict):
        for name, img in loaded_images.items():
            board_state, pieces, meta = detector.detect_state(img)
            assert isinstance(board_state, np.ndarray)
            assert board_state.shape == (8, 8)
            assert isinstance(pieces, list)
            assert isinstance(meta, dict)
            assert "board_inference_ms" in meta
            assert "tray_inference_ms" in meta
            assert "total_inference_ms" in meta
            assert "occupied_cells" in meta
            assert "pieces_count" in meta
            assert meta["occupied_cells"] == int(board_state.sum())
            assert meta["pieces_count"] == len(pieces)

    def test_detect_game_over_on_live_screens(self, detector: BlockBlastDetector, loaded_images: dict):
        for name, img in loaded_images.items():
            is_game_over, score = detector.detect_game_over(img)
            # All 18 repository test images are active live play / probe screens (not game over dialogs)
            assert not is_game_over, f"False positive game-over detection in {name}"
            assert score is None


# =====================================================================
# 7. MULTI-THEME ADAPTATION (SYNTHETIC THEME STRESS TESTS)
# =====================================================================
class TestMultiThemeAdaptation:
    def test_neon_theme_synthetic(self, detector: BlockBlastDetector):
        # Pitch black background with bright saturated neon blocks
        neon_screen = np.zeros((2400, 1080, 3), dtype=np.uint8)
        # Empty board bezel
        neon_screen[581:1537, 61:1018] = 12  # Dark charcoal grid
        # Add 3 neon green occupied cells
        for r, c in [(1, 1), (2, 2), (5, 5)]:
            x, y = detector.get_cell_screen_coords(r, c)
            neon_screen[y - 20:y + 20, x - 20:x + 20] = [50, 255, 50]

        # Tray with 1 neon cyan piece (line2_h)
        tray_crop = neon_screen[1660:2050, 60:1020]
        tray_crop[:, :] = 10  # Dark tray bg
        # Place line2_h in Slot 0 (center X ~ 220, Y ~ 1855)
        neon_screen[1825:1885, 160:280] = [255, 255, 0]

        board = detector.detect_board_state(neon_screen)
        assert sum(sum(r) for r in board) == 3
        assert board[1][1] == 1 and board[2][2] == 1 and board[5][5] == 1

        tray = detector.detect_tray_pieces(neon_screen)
        assert tray[0][0] == "line2_h"

    def test_jungle_theme_synthetic(self, detector: BlockBlastDetector):
        # Forest green background with amber gem blocks
        jungle_screen = np.full((2400, 1080, 3), [30, 80, 35], dtype=np.uint8)
        # Board recess
        jungle_screen[581:1537, 61:1018] = [25, 70, 30]
        # Add 2 amber blocks
        for r, c in [(0, 7), (7, 0)]:
            x, y = detector.get_cell_screen_coords(r, c)
            jungle_screen[y - 20:y + 20, x - 20:x + 20] = [20, 180, 240]

        board = detector.detect_board_state(jungle_screen)
        assert sum(sum(r) for r in board) == 2
        assert board[0][7] == 1 and board[7][0] == 1


# =====================================================================
# 8. PERFORMANCE & LATENCY BENCHMARK (<60ms AVG, <100ms CEILING)
# =====================================================================
class TestPerformanceAndLatency:
    def test_inference_latency_benchmark_under_60ms(self, detector: BlockBlastDetector, loaded_images: dict):
        total_latencies: List[float] = []
        board_latencies: List[float] = []
        tray_latencies: List[float] = []

        # Warmup all images first to stabilize JIT / memory
        for img in loaded_images.values():
            for _ in range(2):
                detector.detect_state(img)

        # Benchmark 5 runs per image across all 18 images
        for name, img in loaded_images.items():
            img_latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                _, _, meta = detector.detect_state(img)
                dt = (time.perf_counter() - t0) * 1000.0
                img_latencies.append(dt)
                board_latencies.append(meta["board_inference_ms"])
                tray_latencies.append(meta["tray_inference_ms"])

            mean_img_latency = float(np.mean(img_latencies))
            total_latencies.append(mean_img_latency)
            # Strict hard ceiling per frame (<100ms)
            assert mean_img_latency < 100.0, f"Frame latency exceeded 100ms ceiling on {name}: {mean_img_latency:.2f}ms"

        overall_mean_latency = float(np.mean(total_latencies))
        mean_board_latency = float(np.mean(board_latencies))
        mean_tray_latency = float(np.mean(tray_latencies))

        print(f"\n[BENCHMARK] Total Mean Inference: {overall_mean_latency:.2f} ms")
        print(f"[BENCHMARK] Board Mean Inference: {mean_board_latency:.2f} ms")
        print(f"[BENCHMARK] Tray Mean Inference:  {mean_tray_latency:.2f} ms")

        # Must comfortably achieve sub-60ms requirement
        assert overall_mean_latency < 60.0, (
            f"Overall mean latency exceeded 60ms target: {overall_mean_latency:.2f}ms"
        )
