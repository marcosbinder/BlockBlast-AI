"""
Unit & Integration Tests for FastADBSocketClient and ADB Infrastructure.

Tests cover:
- Physical coordinate bounds clamping & validation (1080x2400)
- Tray cancellation zone hazard detection & safe release clamping (Y <= 1580)
- ADB protocol request framing (4-byte hex length prefix)
- PNG byte stream cleaning & magic header recovery
- Mock TCP ADB server communication (transport switch, exec commands, error handling)
- Screencap capture via socket and subprocess fallback
- Touch gesture dispatching (swipe, tap) with automatic clamping
- Latency tracking and connection status reporting
"""

import os
import socket
import struct
import subprocess
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import adb_client
from adb_client import (
    BOARD_BOTTOM_Y,
    BOARD_LEFT_X,
    BOARD_RIGHT_X,
    BOARD_TOP_Y,
    MAX_SAFE_RELEASE_Y,
    PNG_MAGIC_HEADER,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TRAY_CANCEL_ZONE_START_Y,
    FastADBSocketClient,
    clamp_coordinate_x,
    clamp_coordinate_y,
    clamp_coordinates,
    clamp_release_y,
    clean_png_bytes,
    find_adb_executable,
    is_in_tray_cancel_zone,
    is_within_bounds,
)


class MockADBServer:
    """A lightweight in-process TCP server simulating the ADB daemon protocol for testing."""

    def __init__(self, serial="ZF524K4RCM", response_payload=b"", should_fail_transport=False, should_fail_cmd=False):
        self.serial = serial
        self.response_payload = response_payload
        self.should_fail_transport = should_fail_transport
        self.should_fail_cmd = should_fail_cmd
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(("127.0.0.1", 0))
        self.port = self.server_sock.getsockname()[1]
        self.server_sock.listen(1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.server_sock.settimeout(2.0)
            conn, _ = self.server_sock.accept()
            with conn:
                # 1. Read transport request length + string
                len_hex = conn.recv(4).decode("ascii")
                req_len = int(len_hex, 16)
                transport_req = conn.recv(req_len).decode("utf-8")

                if self.should_fail_transport:
                    msg = "device not found"
                    conn.sendall(b"FAIL" + f"{len(msg):04x}".encode("ascii") + msg.encode("utf-8"))
                    return

                conn.sendall(b"OKAY")

                # 2. Read command length + string
                len_hex2 = conn.recv(4).decode("ascii")
                req_len2 = int(len_hex2, 16)
                cmd_req = conn.recv(req_len2).decode("utf-8")

                if self.should_fail_cmd:
                    msg = "command failed"
                    conn.sendall(b"FAIL" + f"{len(msg):04x}".encode("ascii") + msg.encode("utf-8"))
                    return

                conn.sendall(b"OKAY")

                # 3. Stream payload
                if self.response_payload:
                    conn.sendall(self.response_payload)
        except Exception:
            pass
        finally:
            self.server_sock.close()

    def stop(self):
        self.running = False
        try:
            self.server_sock.close()
        except Exception:
            pass


class TestCoordinateBoundsAndClamping(unittest.TestCase):
    """Test suite for coordinate math, safe boundaries, and tray cancellation clamping."""

    def test_clamp_coordinate_x(self):
        self.assertEqual(clamp_coordinate_x(0), 0)
        self.assertEqual(clamp_coordinate_x(540), 540)
        self.assertEqual(clamp_coordinate_x(1080), 1080)
        self.assertEqual(clamp_coordinate_x(-50), 0)
        self.assertEqual(clamp_coordinate_x(1500), 1080)
        self.assertEqual(clamp_coordinate_x(540.6), 541)

    def test_clamp_coordinate_y(self):
        self.assertEqual(clamp_coordinate_y(0), 0)
        self.assertEqual(clamp_coordinate_y(1200), 1200)
        self.assertEqual(clamp_coordinate_y(2400), 2400)
        self.assertEqual(clamp_coordinate_y(-100), 0)
        self.assertEqual(clamp_coordinate_y(3000), 2400)
        self.assertEqual(clamp_coordinate_y(999.4), 999)

    def test_clamp_coordinates_tuple(self):
        self.assertEqual(clamp_coordinates(500, 1000), (500, 1000))
        self.assertEqual(clamp_coordinates(-10, 2500), (0, 2400))
        self.assertEqual(clamp_coordinates(1200, -50), (1080, 0))

    def test_clamp_release_y_safe_ceiling(self):
        # Coordinates above board or in board area remain unchanged
        self.assertEqual(clamp_release_y(600), 600)
        self.assertEqual(clamp_release_y(1200), 1200)
        self.assertEqual(clamp_release_y(1580), 1580)

        # Coordinates inside or beyond tray cancel hazard zone (>=1600) must be clamped to 1580
        self.assertEqual(clamp_release_y(1600), 1580)
        self.assertEqual(clamp_release_y(1677), 1580)
        self.assertEqual(clamp_release_y(1855), 1580)
        self.assertEqual(clamp_release_y(2400), 1580)
        self.assertEqual(clamp_release_y(3000), 1580)

        # Negative values clamped to 0
        self.assertEqual(clamp_release_y(-20), 0)

    def test_is_within_bounds(self):
        self.assertTrue(is_within_bounds(0, 0))
        self.assertTrue(is_within_bounds(1080, 2400))
        self.assertTrue(is_within_bounds(540, 1200))
        self.assertFalse(is_within_bounds(-1, 500))
        self.assertFalse(is_within_bounds(1081, 500))
        self.assertFalse(is_within_bounds(500, -1))
        self.assertFalse(is_within_bounds(500, 2401))

    def test_is_in_tray_cancel_zone(self):
        self.assertFalse(is_in_tray_cancel_zone(580))
        self.assertFalse(is_in_tray_cancel_zone(1537))
        self.assertFalse(is_in_tray_cancel_zone(1580))
        self.assertFalse(is_in_tray_cancel_zone(1599))
        self.assertTrue(is_in_tray_cancel_zone(1600))
        self.assertTrue(is_in_tray_cancel_zone(1855))
        self.assertTrue(is_in_tray_cancel_zone(2400))


class TestADBProtocolFormatting(unittest.TestCase):
    """Test suite for ADB protocol framing and byte sanitization."""

    def test_format_adb_request_str(self):
        req = FastADBSocketClient._format_adb_request("host:transport:ZF524K4RCM")
        expected = b"0019host:transport:ZF524K4RCM"
        self.assertEqual(req, expected)

    def test_format_adb_request_bytes(self):
        req = FastADBSocketClient._format_adb_request(b"exec:screencap -p")
        expected = b"0011exec:screencap -p"
        self.assertEqual(req, expected)

    def test_format_adb_request_hex_length(self):
        # 1-byte command -> "0001"
        self.assertTrue(FastADBSocketClient._format_adb_request("a").startswith(b"0001"))
        # 256-byte payload -> "0100"
        long_payload = "x" * 256
        self.assertTrue(FastADBSocketClient._format_adb_request(long_payload).startswith(b"0100"))

    def test_clean_png_bytes(self):
        # Clean PNG starting with valid magic header
        valid_png = PNG_MAGIC_HEADER + b"\x00\x00\x00\rIHDR..."
        self.assertEqual(clean_png_bytes(valid_png), valid_png)

        # Empty or None
        self.assertEqual(clean_png_bytes(b""), b"")

        # Corrupted CRLF \r\r\n
        corrupted_crlf = b"\x89PNG\r\r\n\x1a\r\n\x00\x00\x00\rIHDR"
        cleaned = clean_png_bytes(corrupted_crlf)
        self.assertTrue(cleaned.startswith(PNG_MAGIC_HEADER))

        # Leading noise before PNG header
        with_leading_noise = b"Some adb warning text\r\n" + valid_png
        cleaned_noise = clean_png_bytes(with_leading_noise)
        self.assertEqual(cleaned_noise, valid_png)


class TestFastADBSocketClientWithMockServer(unittest.TestCase):
    """Test FastADBSocketClient socket communication against simulated local mock server."""

    def test_execute_socket_command_success(self):
        mock_payload = b"Hello from ADB daemon"
        server = MockADBServer(response_payload=mock_payload)
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")

        result = client.execute_socket_command("exec:echo test")
        self.assertEqual(result, mock_payload)

    def test_transport_failure_raises_runtime_error(self):
        server = MockADBServer(should_fail_transport=True)
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")

        with self.assertRaises(RuntimeError) as ctx:
            client.execute_socket_command("exec:echo test")
        self.assertIn("device not found", str(ctx.exception))

    def test_command_failure_raises_runtime_error(self):
        server = MockADBServer(should_fail_cmd=True)
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")

        with self.assertRaises(RuntimeError) as ctx:
            client.execute_socket_command("exec:invalid_command")
        self.assertIn("command failed", str(ctx.exception))

    def test_screencap_socket_success(self):
        # Create a synthetic 1080x2400 dummy PNG image
        dummy_img = np.zeros((2400, 1080, 3), dtype=np.uint8)
        dummy_img[581:1537, 61:1018] = [47, 71, 119]  # fill board region with slate blue
        _, png_bytes = cv2.imencode(".png", dummy_img)
        raw_png = png_bytes.tobytes()

        server = MockADBServer(response_payload=raw_png)
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")

        img = client.screencap_cv2()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (2400, 1080, 3))
        self.assertEqual(client.last_capture_method, "socket")
        self.assertGreater(client.last_screencap_latency_ms, 0.0)

    def test_shell_socket_success(self):
        mock_payload = b"socket shell output\n"
        server = MockADBServer(response_payload=mock_payload)
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")

        out = client.shell("some cmd")
        self.assertEqual(out, "socket shell output\n")

    def test_is_connected_socket_success(self):
        server = MockADBServer()
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM")
        self.assertTrue(client.is_connected())

    def test_socket_premature_eof_raises(self):
        # Create server that immediately closes socket on connect
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        def close_immediately():
            try:
                conn, _ = server.accept()
                conn.close()
            except Exception:
                pass
            finally:
                server.close()

        threading.Thread(target=close_immediately, daemon=True).start()

        client = FastADBSocketClient(host="127.0.0.1", port=port, serial="ZF524K4RCM", connect_timeout=0.1, socket_timeout=0.2)
        with self.assertRaises(Exception):
            client.execute_socket_command("exec:test")


class TestFastADBSocketClientMethodsAndFallback(unittest.TestCase):
    """Test client methods (swipe, tap, subprocess fallback, error handling)."""

    def setUp(self):
        self.client = FastADBSocketClient(
            host="127.0.0.1",
            port=59999,
            serial="ZF524K4RCM",
            connect_timeout=0.05,
            socket_timeout=0.1,
        )

    @patch.object(FastADBSocketClient, "shell")
    def test_tap_dispatches_clamped_coordinates(self, mock_shell):
        mock_shell.return_value = ""

        # Normal tap
        res = self.client.tap(500, 1000)
        self.assertTrue(res)
        mock_shell.assert_called_with("input tap 500 1000")

        # Out of bounds tap
        self.client.tap(-50, 3000)
        mock_shell.assert_called_with("input tap 0 2400")

    @patch.object(FastADBSocketClient, "shell")
    def test_swipe_dispatches_with_safe_clamping(self, mock_shell):
        mock_shell.return_value = ""

        # Swipe with tray cancel zone destination (Y=1855) and clamp_safe_release=True (default)
        res = self.client.swipe(220, 1855, 540, 1855, duration_ms=250, clamp_safe_release=True)
        self.assertTrue(res)
        mock_shell.assert_called_with("input swipe 220 1855 540 1580 250")

        # Swipe with clamp_safe_release=False
        self.client.swipe(220, 1855, 540, 1855, duration_ms=250, clamp_safe_release=False)
        mock_shell.assert_called_with("input swipe 220 1855 540 1855 250")

        # Out of bounds swipe clamped
        self.client.swipe(-100, 500, 1200, 2500, duration_ms=300, clamp_safe_release=False)
        mock_shell.assert_called_with("input swipe 0 500 1080 2400 300")

    @patch("subprocess.run")
    def test_screencap_fallback_to_subprocess_when_socket_fails(self, mock_subproc):
        # Create a synthetic 100x100 dummy PNG image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", dummy_img)
        raw_png = png_bytes.tobytes()

        mock_proc = MagicMock()
        mock_proc.stdout = raw_png
        mock_subproc.return_value = mock_proc

        # Fake adb_path so fallback proceeds
        self.client.adb_path = r"C:\fake\adb.exe"

        # Socket will fail because port 59999 has no listener
        img = self.client.screencap_cv2()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (100, 100, 3))
        self.assertEqual(self.client.last_capture_method, "subprocess")
        self.assertGreater(self.client.last_screencap_latency_ms, 0.0)

    def test_screencap_all_fail_raises_runtime_error(self):
        # No listener and no adb_path
        self.client.adb_path = None
        with self.assertRaises(RuntimeError) as ctx:
            self.client.screencap_cv2()
        self.assertIn("All screencap methods failed", str(ctx.exception))

    @patch("subprocess.run")
    def test_is_connected_subprocess_check(self, mock_subproc):
        self.client.adb_path = r"C:\fake\adb.exe"
        mock_proc = MagicMock()
        mock_proc.stdout = "List of devices attached\nZF524K4RCM\tdevice\n\n"
        mock_subproc.return_value = mock_proc

        self.assertTrue(self.client.is_connected())

        mock_proc.stdout = "List of devices attached\n\n"
        self.assertFalse(self.client.is_connected())

    def test_find_adb_executable(self):
        # Should return a string path if found or None
        result = find_adb_executable()
        if result is not None:
            self.assertTrue(isinstance(result, str))
            self.assertTrue(os.path.isfile(result))

    @patch("subprocess.run")
    def test_shell_subprocess_fallback(self, mock_subproc):
        self.client.adb_path = r"C:\fake\adb.exe"
        mock_proc = MagicMock()
        mock_proc.stdout = "dummy output\n"
        mock_subproc.return_value = mock_proc

        out = self.client.shell("echo test")
        self.assertEqual(out, "dummy output\n")


class TestSocketEdgeCases(unittest.TestCase):
    """Test edge cases for socket streaming, corrupted frames, and timeouts."""

    def test_screencap_socket_corrupted_png_fallback_to_subprocess(self):
        # Server sends garbage bytes that are not a valid PNG
        server = MockADBServer(response_payload=b"INVALID_PNG_BYTES")
        client = FastADBSocketClient(host="127.0.0.1", port=server.port, serial="ZF524K4RCM", connect_timeout=0.1, socket_timeout=0.2)
        client.adb_path = r"C:\fake\adb.exe"

        with patch("subprocess.run") as mock_subproc:
            dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
            _, png_bytes = cv2.imencode(".png", dummy_img)
            mock_proc = MagicMock()
            mock_proc.stdout = png_bytes.tobytes()
            mock_subproc.return_value = mock_proc

            img = client.screencap_cv2()
            self.assertEqual(img.shape, (50, 50, 3))
            self.assertEqual(client.last_capture_method, "subprocess")


if __name__ == "__main__":
    unittest.main(verbosity=2)
