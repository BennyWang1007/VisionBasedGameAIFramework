"""
Windows API utilities for screenshot capture and input control.
"""
import win32gui
import win32ui
import win32con
import win32api
import numpy as np
from typing import Optional
import time
import ctypes
import ctypes.wintypes

# Enable DPI awareness to get correct window sizes on high-DPI displays
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # Fallback for older Windows
    except Exception:
        pass


class WindowsGameController:
    """Controller for capturing screenshots and sending inputs to game windows"""

    def __init__(self, window_name: Optional[str] = None, window_class: Optional[str] = None):
        """
        Initialize the controller.

        Args:
            window_name: Name of the game window (partial match)
            window_class: Class name of the window (optional, more precise)
        """
        self.window_name = window_name
        self.window_class = window_class
        self.hwnd: Optional[int] = None

        if window_name or window_class:
            self.find_window()

    def find_window(self) -> bool:
        """
        Find and store the game window handle.

        Returns:
            True if window found, False otherwise
        """
        if self.window_class:
            self.hwnd = win32gui.FindWindow(self.window_class, None)
        elif self.window_name:
            def callback(hwnd, windows_list):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if self.window_name.lower() in title.lower():
                        windows_list.append(hwnd)
                return True

            windows: list[int] = []
            win32gui.EnumWindows(callback, windows)
            if windows:
                self.hwnd = windows[0]

        return self.hwnd is not None

    def capture_screenshot(self, region: Optional[tuple[int, int, int, int]] = None, use_printwindow: bool = True) -> np.ndarray:
        """
        Capture screenshot from the game window.

        Args:
            region: Optional (x, y, width, height) region to capture.
                   If None, captures entire window.
            use_printwindow: If True, use PrintWindow API which works for background windows.
                            If False, use BitBlt from desktop (works for hardware-accelerated games).

        Returns:
            Screenshot as numpy array (H, W, 3) in RGB format
        """
        if not self.hwnd:
            raise RuntimeError("Window not found. Call find_window() first.")

        # Get window dimensions using GetWindowRect
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # For DPI scaling issues, try to get the actual bitmap size from DWM
        try:
            # Use DwmGetWindowAttribute to get the extended frame bounds (actual size)
            rect = ctypes.wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            ctypes.windll.dwmapi.DwmGetWindowAttribute(
                ctypes.wintypes.HWND(self.hwnd),
                ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
                ctypes.byref(rect),
                ctypes.sizeof(rect)
            )
            # Use the DWM bounds if they give a larger size
            dwm_width = rect.right - rect.left
            dwm_height = rect.bottom - rect.top
            if dwm_width > width:
                width = dwm_width
                left = rect.left
            if dwm_height > height:
                height = dwm_height
                top = rect.top
        except Exception:
            pass  # DWM not available, use original dimensions

        if region:
            x, y, w, h = region
        else:
            x, y, w, h = 0, 0, width, height

        # Create device contexts and bitmap
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        if use_printwindow:
            # Use PrintWindow - works for background/occluded windows
            # PW_RENDERFULLCONTENT = 2 (captures DirectX content on Windows 8.1+)
            try:
                result = ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 2)
                if result == 0:
                    # Fallback to default PrintWindow (flag = 0 captures full window)
                    ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 0)
            except Exception:
                # Fallback to basic PrintWindow
                win32gui.SendMessage(self.hwnd, win32con.WM_PRINT, save_dc.GetSafeHdc(),
                                     win32con.PRF_CHILDREN | win32con.PRF_CLIENT | win32con.PRF_OWNED)
        else:
            # Use BitBlt from desktop DC - works for hardware-accelerated games but window must be visible
            desktop_dc = win32gui.GetDC(0)
            desktop_mfc_dc = win32ui.CreateDCFromHandle(desktop_dc)
            save_dc.BitBlt((0, 0), (width, height), desktop_mfc_dc, (left, top), win32con.SRCCOPY)
            desktop_mfc_dc.DeleteDC()
            win32gui.ReleaseDC(0, desktop_dc)

        # Convert to numpy array
        signed_ints_array = bitmap.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (height, width, 4)

        # Clean up
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        # Handle region cropping if specified
        if region:
            img = img[y:y+h, x:x+w]

        # Convert BGRA to RGB
        return img[:, :, :3][:, :, ::-1].copy()

    def send_key(self, key: str, hold_time: float = 0.05, background: bool = True) -> None:
        """
        Send a key press to the game window (key down + delay + key up).

        Args:
            key: Key to press (e.g., 'w', 'space', 'left')
            hold_time: How long to hold the key (seconds)
            background: If True, send key without stealing focus (using PostMessage)
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        vk_code = self._get_vk_code(key)

        if background:
            # Send key without stealing focus using PostMessage
            # WM_KEYDOWN = 0x0100, WM_KEYUP = 0x0101
            scan_code = win32api.MapVirtualKey(vk_code, 0)
            lparam_down = (scan_code << 16) | 1
            lparam_up = (scan_code << 16) | 0xC0000001  # Key up flag

            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, lparam_down)
            time.sleep(hold_time)
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
        else:
            # Focus the window (old behavior)
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.01)  # Small delay to ensure focus

            # Send key press using keybd_event
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(hold_time)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    def _get_vk_code(self, key: str) -> int:
        """Get virtual key code for a key."""
        key_map = {
            'space': win32con.VK_SPACE,
            'enter': win32con.VK_RETURN,
            'left': win32con.VK_LEFT,
            'right': win32con.VK_RIGHT,
            'up': win32con.VK_UP,
            'down': win32con.VK_DOWN,
            'shift': win32con.VK_SHIFT,
            'ctrl': win32con.VK_CONTROL,
            'alt': win32con.VK_MENU,
            'esc': win32con.VK_ESCAPE,
        }
        if key.lower() in key_map:
            return key_map[key.lower()]
        elif len(key) == 1:
            return ord(key.upper())
        else:
            raise ValueError(f"Unknown key: {key}")

    def key_down(self, key: str, background: bool = True) -> None:
        """
        Send key down event (press without release).

        Args:
            key: Key to press down (e.g., 'w', 'space', 'left')
            background: If True, send key without stealing focus
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        vk_code = self._get_vk_code(key)

        if background:
            scan_code = win32api.MapVirtualKey(vk_code, 0)
            lparam_down = (scan_code << 16) | 1
            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, lparam_down)
        else:
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.01)
            win32api.keybd_event(vk_code, 0, 0, 0)

    def key_up(self, key: str, background: bool = True) -> None:
        """
        Send key up event (release key).

        Args:
            key: Key to release (e.g., 'w', 'space', 'left')
            background: If True, send key without stealing focus
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        vk_code = self._get_vk_code(key)

        if background:
            scan_code = win32api.MapVirtualKey(vk_code, 0)
            lparam_up = (scan_code << 16) | 0xC0000001
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
        else:
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    def mouse_move(self, x: int, y: int, background: bool = True) -> None:
        """
        Move mouse to specific position without clicking.

        Args:
            x: X coordinate relative to window
            y: Y coordinate relative to window
            background: If True, send without stealing focus
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        if background:
            lparam = (y << 16) | x
            win32api.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        else:
            left, top, _, _ = win32gui.GetWindowRect(self.hwnd)
            screen_x = left + x
            screen_y = top + y
            win32api.SetCursorPos((screen_x, screen_y))

    def mouse_down(self, x: int, y: int, button: str = 'left', background: bool = True) -> None:
        """
        Send mouse button down event (press without release).

        Args:
            x: X coordinate relative to window
            y: Y coordinate relative to window
            button: 'left' or 'right'
            background: If True, send without stealing focus
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        if background:
            lparam = (y << 16) | x
            if button == 'left':
                win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
            elif button == 'right':
                win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
        else:
            left, top, _, _ = win32gui.GetWindowRect(self.hwnd)
            screen_x = left + x
            screen_y = top + y
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.01)
            win32api.SetCursorPos((screen_x, screen_y))
            time.sleep(0.01)
            if button == 'left':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            elif button == 'right':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

    def mouse_up(self, x: int, y: int, button: str = 'left', background: bool = True) -> None:
        """
        Send mouse button up event (release button).

        Args:
            x: X coordinate relative to window
            y: Y coordinate relative to window
            button: 'left' or 'right'
            background: If True, send without stealing focus
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        if background:
            lparam = (y << 16) | x
            if button == 'left':
                win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, lparam)
            elif button == 'right':
                win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, lparam)
        else:
            left, top, _, _ = win32gui.GetWindowRect(self.hwnd)
            screen_x = left + x
            screen_y = top + y
            win32api.SetCursorPos((screen_x, screen_y))
            if button == 'left':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif button == 'right':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def send_mouse_click(self, x: int, y: int, button: str = 'left', background: bool = True) -> None:
        """
        Send mouse click to specific position.

        Args:
            x: X coordinate relative to window
            y: Y coordinate relative to window
            button: 'left' or 'right'
            background: If True, send click without stealing focus (using PostMessage)
        """
        if not self.hwnd:
            raise RuntimeError("Window not found.")

        if background:
            # Send mouse click without stealing focus using PostMessage
            # Pack coordinates into lParam
            lparam = (y << 16) | x

            if button == 'left':
                win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
                time.sleep(0.05)
                win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, lparam)
            elif button == 'right':
                win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
                time.sleep(0.05)
                win32api.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, lparam)
        else:
            # Get window position
            left, top, _, _ = win32gui.GetWindowRect(self.hwnd)

            # Convert to screen coordinates
            screen_x = left + x
            screen_y = top + y

            # Focus window
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.01)

            # Move mouse
            win32api.SetCursorPos((screen_x, screen_y))
            time.sleep(0.01)

            # Click
            if button == 'left':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif button == 'right':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def is_window_active(self) -> bool:
        """Check if the game window is active/visible."""
        if not self.hwnd:
            return False
        return bool(win32gui.IsWindowVisible(self.hwnd))

    def get_window_rect(self) -> Optional[tuple[int, int, int, int]]:
        """Get window rectangle (left, top, right, bottom)."""
        if not self.hwnd:
            return None
        return win32gui.GetWindowRect(self.hwnd)
