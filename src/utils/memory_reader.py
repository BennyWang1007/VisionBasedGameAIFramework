"""
Windows process memory reading utilities.
Allows reading game state directly from memory addresses.
"""
import ctypes
from ctypes import wintypes
import struct
from typing import Any, Optional
# import win32api
# import win32con


# Windows API constants
PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_OPERATION = 0x0008

# Memory protection constants
PAGE_EXECUTE_READWRITE = 0x40
PAGE_READWRITE = 0x04

# Kernel32 functions
kernel32 = ctypes.windll.kernel32

OpenProcess = kernel32.OpenProcess
OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
OpenProcess.restype = wintypes.HANDLE

ReadProcessMemory = kernel32.ReadProcessMemory
ReadProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)
]
ReadProcessMemory.restype = wintypes.BOOL

WriteProcessMemory = kernel32.WriteProcessMemory
WriteProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPVOID,
    wintypes.LPCVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)
]
WriteProcessMemory.restype = wintypes.BOOL


class MemoryReader:
    """Read and write game process memory"""

    def __init__(self, process_name: Optional[str] = None, pid: Optional[int] = None):
        """
        Initialize memory reader.

        Args:
            process_name: Name of the process (e.g., "game.exe")
            pid: Process ID (if known)
        """
        self.process_name = process_name
        self.pid = pid
        self.process_handle: Optional[int] = None
        self.base_address: Optional[int] = None

        if process_name:
            self.attach_to_process()

    def attach_to_process(self) -> bool:
        """
        Attach to the game process.

        Returns:
            True if successful, False otherwise
        """
        if self.pid is None and self.process_name:
            self.pid = self._get_pid_by_name(self.process_name)

        if self.pid is None:
            return False

        # Open process with read/write access
        self.process_handle = OpenProcess(
            PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION,
            False,
            self.pid
        )

        if self.process_handle:
            self.base_address = self._get_base_address()
            return True

        return False

    def _get_pid_by_name(self, process_name: str) -> Optional[int]:
        """Get process ID by name"""
        import psutil

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.name().lower() == process_name.lower():
                    return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return None

    def _get_base_address(self) -> Optional[int]:
        """Get base address of the process"""
        if not self.pid:
            return None

        try:
            import psutil
            process = psutil.Process(self.pid)

            # Get first module (main executable) base address
            for module in process.memory_maps():
                if '.exe' in module.path:
                    # Parse address from string like "0x400000-0x500000"
                    addr_str = module.addr.split('-')[0]
                    return int(addr_str, 16)
        except Exception:
            pass

        return None

    def read_int(self, address: int, offset: int = 0) -> Optional[int]:
        """
        Read 4-byte integer from memory.

        Args:
            address: Memory address (can be relative to base)
            offset: Additional offset

        Returns:
            Integer value or None if failed
        """
        data = self.read_bytes(address, offset, 4)
        if data:
            return struct.unpack('<i', data)[0]
        return None

    def read_float(self, address: int, offset: int = 0) -> Optional[float]:
        """Read 4-byte float from memory"""
        data = self.read_bytes(address, offset, 4)
        if data:
            return struct.unpack('<f', data)[0]
        return None

    def read_double(self, address: int, offset: int = 0) -> Optional[float]:
        """Read 8-byte double from memory"""
        data = self.read_bytes(address, offset, 8)
        if data:
            return struct.unpack('<d', data)[0]
        return None

    def read_long(self, address: int, offset: int = 0) -> Optional[int]:
        """Read 8-byte long from memory"""
        data = self.read_bytes(address, offset, 8)
        if data:
            return struct.unpack('<q', data)[0]
        return None

    def read_byte(self, address: int, offset: int = 0) -> Optional[int]:
        """Read 1 byte from memory"""
        data = self.read_bytes(address, offset, 1)
        if data:
            return struct.unpack('<B', data)[0]
        return None

    def read_bool(self, address: int, offset: int = 0) -> Optional[bool]:
        """Read boolean from memory"""
        byte = self.read_byte(address, offset)
        return bool(byte) if byte is not None else None

    def read_string(self, address: int, offset: int = 0, length: int = 256) -> Optional[str]:
        """Read null-terminated string from memory"""
        data = self.read_bytes(address, offset, length)
        if data:
            # Find null terminator
            null_pos = data.find(b'\x00')
            if null_pos != -1:
                data = data[:null_pos]
            try:
                return data.decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                return data.decode('ascii', errors='ignore')
        return None

    def read_bytes(self, address: int, offset: int = 0, size: int = 4) -> Optional[bytes]:
        """
        Read raw bytes from memory.

        Args:
            address: Memory address
            offset: Additional offset
            size: Number of bytes to read

        Returns:
            Bytes read or None if failed
        """
        if not self.process_handle:
            return None

        # Calculate final address
        final_address = address + offset

        # Create buffer
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)

        # Read memory
        success = ReadProcessMemory(
            self.process_handle,
            ctypes.c_void_p(final_address),
            buffer,
            size,
            ctypes.byref(bytes_read)
        )

        if success and bytes_read.value == size:
            return buffer.raw

        return None

    def read_pointer_chain(self, base: int, offsets: list[int]) -> Optional[int]:
        """
        Follow a pointer chain.

        Args:
            base: Base address
            offsets: List of offsets to follow

        Returns:
            Final address or None if chain breaks

        Example:
            # To read [[base + 0x10] + 0x20] + 0x30
            addr = reader.read_pointer_chain(base, [0x10, 0x20, 0x30])
        """
        address = base

        for i, offset in enumerate(offsets):
            if i < len(offsets) - 1:
                # Read pointer
                ptr = self.read_long(address, offset)
                if ptr is None:
                    return None
                address = ptr
            else:
                # Last offset, just add it
                address = address + offset

        return address

    def write_int(self, address: int, value: int, offset: int = 0) -> bool:
        """Write 4-byte integer to memory"""
        data = struct.pack('<i', value)
        return self.write_bytes(address, data, offset)

    def write_float(self, address: int, value: float, offset: int = 0) -> bool:
        """Write 4-byte float to memory"""
        data = struct.pack('<f', value)
        return self.write_bytes(address, data, offset)

    def write_bytes(self, address: int, data: bytes, offset: int = 0) -> bool:
        """Write raw bytes to memory"""
        if not self.process_handle:
            return False

        final_address = address + offset
        bytes_written = ctypes.c_size_t(0)

        success = WriteProcessMemory(
            self.process_handle,
            ctypes.c_void_p(final_address),
            data,
            len(data),
            ctypes.byref(bytes_written)
        )

        return success and bytes_written.value == len(data)

    def close(self) -> None:
        """Close process handle"""
        if self.process_handle:
            kernel32.CloseHandle(self.process_handle)
            self.process_handle = None


class GameMemoryMonitor:
    """High-level interface for monitoring game memory"""

    def __init__(self, process_name: str):
        """
        Initialize memory monitor.

        Args:
            process_name: Name of the game process
        """
        self.reader = MemoryReader(process_name=process_name)
        self.memory_map: dict[str, dict[str, Any]] = {}  # Store known memory addresses

    def register_address(self, name: str, address: int, data_type: str = 'int') -> None:
        """
        Register a memory address for monitoring.

        Args:
            name: Friendly name (e.g., 'player_health')
            address: Memory address
            data_type: 'int', 'float', 'double', 'bool', 'byte'
        """
        self.memory_map[name] = {
            'address': address,
            'type': data_type
        }

    def register_pointer_chain(self, name: str, base: int, offsets: list[int],
                               data_type: str = 'int') -> None:
        """
        Register a pointer chain.

        Args:
            name: Friendly name
            base: Base address
            offsets: List of offsets
            data_type: Data type to read
        """
        self.memory_map[name] = {
            'base': base,
            'offsets': offsets,
            'type': data_type,
            'is_pointer_chain': True
        }

    def read_value(self, name: str) -> int | float | bool | None:
        """
        Read a registered value.

        Args:
            name: Registered name

        Returns:
            Value or None if failed
        """
        if name not in self.memory_map:
            return None

        info = self.memory_map[name]

        # Resolve address
        if info.get('is_pointer_chain'):
            address = self.reader.read_pointer_chain(info['base'], info['offsets'])
            if address is None:
                return None
        else:
            address = info['address']

        assert address is not None, "Address should not be None"

        # Read based on type
        data_type = info['type']
        if data_type == 'int':
            return self.reader.read_int(address)
        elif data_type == 'float':
            return self.reader.read_float(address)
        elif data_type == 'double':
            return self.reader.read_double(address)
        elif data_type == 'bool':
            return self.reader.read_bool(address)
        elif data_type == 'byte':
            return self.reader.read_byte(address)

        return None

    def read_all(self) -> dict[str, int | float | bool | None]:
        """
        Read all registered values.

        Returns:
            Dictionary of name -> value
        """
        return {
            name: self.read_value(name)
            for name in self.memory_map.keys()
        }

    def close(self) -> None:
        """Clean up"""
        self.reader.close()
