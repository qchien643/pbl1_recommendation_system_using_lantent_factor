#!/bin/bash
# launch_multi.sh — Mo 1 server + N client trong cac cua so Windows Terminal moi
# Ban cross-platform cho Git Bash (Windows) dung Windows Terminal (`wt`) neu co.
# Usage: bash tools/launch_multi.sh [n=3]
#
# Linux/macOS tuong duong: dung gnome-terminal, iTerm, etc. — nhung
# don gian nhat van la mo tay tung terminal va chay lenh ./build/client ...

set -e
cd "$(dirname "$0")/.."

N=${1:-3}

if [ ! -f build/server.exe ] && [ ! -f build/server ]; then
  echo "[ERROR] build/server.exe khong ton tai."
  echo "Chay: cmake -S . -B build && cmake --build build"
  exit 1
fi

# Windows: dung Windows Terminal neu co, fallback sang start
if command -v wt.exe >/dev/null 2>&1; then
  echo "[Launcher] Using Windows Terminal — 1 server + $N clients"
  CMD="wt.exe -w 0 new-tab --title SERVER -d \"$(pwd -W 2>/dev/null || pwd)\" cmd /k \"build\\server.exe --server\""
  for i in $(seq 1 $N); do
    CMD="$CMD ; new-tab --title \"CLIENT $i\" -d \"$(pwd -W 2>/dev/null || pwd)\" cmd /k \"timeout /t 2 && build\\client.exe --client 127.0.0.1 $i\""
  done
  eval "$CMD"
  echo "[Launcher] Sang tab SERVER nhap 1234 de mo ca."
  exit 0
fi

# Fallback: cmd start cho Windows
if [ -n "$WINDIR" ] || [ -n "$MSYSTEM" ]; then
  cmd.exe //c "tools\\launch_multi.bat $N"
  exit 0
fi

# Linux/macOS: khong ro terminal emulator, bao user mo tay
echo "[Launcher] He dieu hanh khong phai Windows. Mo $((N+1)) terminal tay va chay:"
echo "  Terminal 1 (SERVER): ./build/server --server"
for i in $(seq 1 $N); do
  echo "  Terminal $((i+1)) (CLIENT $i): ./build/client --client 127.0.0.1 $i"
done
