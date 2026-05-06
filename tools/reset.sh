#!/bin/bash
# reset.sh — Xóa toàn bộ dữ liệu runtime + reseed về fresh state.
#
# Xóa:
#   data/*.tbl                   (tables mới)
#   data/*.dat                   (legacy, nếu còn)
#   data/transactions.log
#   data/personas.txt
#   data/reports/*.txt
#
# Giữ lại:
#   data/menu.txt                (input file, không bị reset)
#
# Sau đó chạy seed_data để sinh lại 10 personas + ~180 transactions + LFM trained.
#
# Usage:
#   bash tools/reset.sh
#   bash tools/reset.sh --no-seed   # chỉ xóa, không seed lại

set -e
cd "$(dirname "$0")/.."

NOSEED=0
if [ "$1" = "--no-seed" ]; then NOSEED=1; fi

echo "════════════════════════════════════════════════════════════"
echo "  Reset PBL1 data → fresh state"
echo "════════════════════════════════════════════════════════════"

# -------- Kill running server nếu có --------
if pgrep -f "server.exe" >/dev/null 2>&1; then
    echo "▶ Dừng server.exe đang chạy..."
    taskkill //F //IM server.exe >/dev/null 2>&1 || true
fi

# -------- Xóa data --------
echo "▶ Xóa data files..."
rm -f data/*.tbl
rm -f data/*.dat
rm -f data/transactions.log
rm -f data/personas.txt
rm -f data/reports/*.txt 2>/dev/null || true
echo "  ✓ Đã xóa: *.tbl, *.dat, transactions.log, personas.txt, reports/*.txt"

if [ "$NOSEED" = "1" ]; then
    echo ""
    echo "  ✅ Reset xong. (--no-seed) → không reseed."
    exit 0
fi

# -------- Reseed --------
echo ""
echo "▶ Reseed dữ liệu mẫu..."
if [ ! -f "build/seed_data.exe" ]; then
    echo "  → build/seed_data.exe chưa có. Chạy build..."
    if [ -n "$MSYSTEM" ]; then
        cmake -S . -B build -G "MinGW Makefiles" >/dev/null
    else
        cmake -S . -B build >/dev/null
    fi
    cmake --build build --target seed_data
fi

./build/seed_data.exe | tail -5

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Reset + reseed xong. Chạy: bash tools/run.sh"
echo "════════════════════════════════════════════════════════════"
