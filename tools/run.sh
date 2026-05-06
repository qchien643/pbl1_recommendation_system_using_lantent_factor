#!/bin/bash
# run.sh — One-shot bootstrap + launch React Ink UI trên Windows (Git Bash).
#
# Tự động:
#   1. Build C++ binaries (cmake) nếu chưa
#   2. Seed dữ liệu mẫu (.tbl tables) nếu data/users.tbl chưa tồn tại
#   3. Cài cli/node_modules nếu chưa
#   4. Mở Server Dashboard (React Ink/blessed-contrib) + N Client UI (React Ink)
#      mỗi cái trong 1 cửa sổ riêng (Windows Terminal nếu có, fallback `cmd start`)
#
# Usage:
#   bash tools/run.sh                # 3 clients (default)
#   bash tools/run.sh 5              # 5 clients
#
# Yêu cầu: Git Bash + g++ + cmake + node 18+ trên Windows.

set -e

# -------- Move to project root --------
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ROOT_WIN="$(pwd -W 2>/dev/null || pwd)"   # Win-style path cho cmd.exe

N=${1:-3}

# -------- Sanity checks --------
need() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Thiếu: $1"; exit 1; }; }
need cmake
need node
need npm
command -v g++ >/dev/null 2>&1 || command -v cl >/dev/null 2>&1 || { echo "❌ Thiếu C++ compiler (g++ hoặc MSVC)"; exit 1; }

echo "════════════════════════════════════════════════════════════"
echo "  PBL1 Restaurant — Auto launcher (Server + $N Clients)"
echo "════════════════════════════════════════════════════════════"

# -------- 1/4: Build --------
echo ""
echo "▶ [1/4] Build C++ binaries"
if [ ! -d build ]; then
    if [ -n "$MSYSTEM" ]; then
        cmake -S . -B build -G "MinGW Makefiles"
    else
        cmake -S . -B build
    fi
fi
cmake --build build

# Verify binaries
for bin in server.exe client.exe seed_data.exe; do
    if [ ! -f "build/$bin" ]; then
        echo "❌ build/$bin không tồn tại sau khi build"
        exit 1
    fi
done
echo "  ✓ server.exe, client.exe, seed_data.exe sẵn sàng"

# -------- 2/4: Seed data --------
echo ""
echo "▶ [2/4] Seed dữ liệu mẫu"
if [ -f "data/users.tbl" ] && [ -f "data/transactions.tbl" ] && [ -f "data/lfm_p.tbl" ]; then
    echo "  ✓ data/*.tbl đã có (10 personas) — bỏ qua seed"
else
    echo "  → Sinh data: 10 personas + ~180 transactions + train LFM..."
    ./build/seed_data.exe | tail -3
    echo "  ✓ Seed xong"
fi

# -------- 3/4: npm deps --------
echo ""
echo "▶ [3/4] Cài npm deps cho React Ink UI"
if [ -d "cli/node_modules/ink" ]; then
    echo "  ✓ cli/node_modules đã có — bỏ qua npm install"
else
    echo "  → cd cli && npm install (lần đầu, ~30s)..."
    (cd cli && npm install --silent)
    echo "  ✓ npm install xong"
fi

# -------- 4/4: Launch --------
echo ""
echo "▶ [4/4] Spawn Server Dashboard + $N Client UI"

# cygpath bắt buộc — convert đường dẫn POSIX → Windows
if ! command -v cygpath >/dev/null 2>&1; then
    echo "❌ Thiếu cygpath (Git Bash chuẩn có sẵn)"; exit 1
fi

CLI_DIR_WIN=$(cygpath -w "$ROOT/cli")

# Wrapper .bat đặt tại tools/_runtmp/ (path ổn định + Windows-friendly)
TMPDIR="$ROOT/tools/_runtmp"
mkdir -p "$TMPDIR"
rm -f "$TMPDIR"/*.bat 2>/dev/null || true

SERVER_BAT="$TMPDIR/start_server.bat"
cat > "$SERVER_BAT" <<EOF
@echo off
title PBL1 Server Dashboard
cd /d "$CLI_DIR_WIN"
cls
call npm run server --silent
echo.
echo Server exited. Press any key to close.
pause >nul
EOF

for i in $(seq 1 "$N"); do
    CBAT="$TMPDIR/start_client_$i.bat"
    cat > "$CBAT" <<EOF
@echo off
title PBL1 Client UI $i
cd /d "$CLI_DIR_WIN"
ping -n 5 127.0.0.1 >nul
cls
call npm start --silent -- 127.0.0.1 $i
echo.
echo Client exited. Press any key to close.
pause >nul
EOF
done

SERVER_BAT_WIN=$(cygpath -w "$SERVER_BAT")

if command -v wt.exe >/dev/null 2>&1; then
    echo "  → Dùng Windows Terminal (wt.exe) — gộp tabs"
    wt.exe -w 0 new-tab --title "SERVER" "$SERVER_BAT_WIN"
    sleep 4
    for i in $(seq 1 "$N"); do
        CBAT_WIN=$(cygpath -w "$TMPDIR/start_client_$i.bat")
        wt.exe -w 0 new-tab --title "CLIENT-$i" "$CBAT_WIN"
        sleep 1
    done
else
    echo "  → Dùng cmd start (mỗi UI 1 window riêng)"
    cmd.exe //c "start \"SERVER UI\" \"$SERVER_BAT_WIN\""
    sleep 4
    for i in $(seq 1 "$N"); do
        CBAT_WIN=$(cygpath -w "$TMPDIR/start_client_$i.bat")
        cmd.exe //c "start \"CLIENT UI $i\" \"$CBAT_WIN\""
        sleep 1
    done
fi

echo "  ℹ Wrapper batches: $TMPDIR/"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Đã mở: 1 Server Dashboard + $N Client UI"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  📋 Hướng dẫn:"
echo "     • Server Dashboard: gõ 1234 + Enter để MỞ CA → broadcast START"
echo "     • Mỗi Client UI: nhập SDT 10 chữ số."
echo "       SDT mẫu (data/personas.txt):"
echo "         0901234567 = Anh Nam   (gợi ý: Trà Đá + Phở Bò)"
echo "         0923456789 = Bác Hùng  (gợi ý: Bún Bò + Chè)"
echo "         0956789012 = Cô Tư     (gợi ý: Phở Gà + Chè)"
echo "     • Đóng ca: nhập lại 1234 ở Server Dashboard."
echo ""
echo "  🔄 Reset về fresh state: bash tools/reset.sh"
echo "════════════════════════════════════════════════════════════"
