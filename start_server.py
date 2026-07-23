import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")


def find_process_using_port(port: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((HOST, port))
            return None
    except OSError:
        return True


def clear_port(port: int):
    if os.name == "nt":
        result = subprocess.run(
            ["powershell", "-Command", f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    return False


if __name__ == "__main__":
    if find_process_using_port(PORT):
        clear_port(PORT)
        time.sleep(1)
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT)], check=True)
