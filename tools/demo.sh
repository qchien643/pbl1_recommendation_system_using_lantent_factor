#!/bin/bash
# demo.sh — Kich ban "Mot ca lam viec tai nha hang"
# Chay tu goc project: bash tools/demo.sh

set -e
cd "$(dirname "$0")/.."

hr() { echo "======================================================"; }
section() { echo; hr; echo "  $*"; hr; }

# -----------------------------------------------------------------------------
# Chuan bi
# -----------------------------------------------------------------------------
section "CHUAN BI: Reset data + re-seed (10 personas, LFM da train)"
rm -f data/lfm_P.dat data/lfm_Q.dat data/users.dat data/reports/*.txt 2>/dev/null || true
./build/seed_data.exe | tail -15

# -----------------------------------------------------------------------------
# Mo ca
# -----------------------------------------------------------------------------
section "SERVER: Thu ngan mo ca ma so 1234, cho khach"

# Server roi vao background, duoc feed 2 lan "1234" (mo + dong ca)
( echo "1234"; sleep 25; echo "1234" ) | ./build/server.exe --server > /tmp/demo_server.log 2>&1 &
SPID=$!
sleep 1.2

# -----------------------------------------------------------------------------
# Ham: mot khach ghe ban (dung client --json, gui scripted cmds)
# -----------------------------------------------------------------------------
one_customer() {
  local label="$1" phone="$2" banner="$3" items_script="$4"
  section "KHACH GHE: $label  (SDT: $phone)"
  echo "  $banner"
  echo
  (
    echo "{\"cmd\":\"login\",\"phone\":\"$phone\"}"
    sleep 0.4
    # items_script la chuoi "CODE1 QTY1;CODE2 QTY2;..."
    IFS=';' read -ra PAIRS <<< "$items_script"
    for pair in "${PAIRS[@]}"; do
      code="${pair% *}"
      qty="${pair##* }"
      echo "{\"cmd\":\"add_item\",\"code\":\"$code\",\"qty\":$qty}"
      sleep 0.3
    done
    echo '{"cmd":"finish"}'
    sleep 0.3
    echo '{"cmd":"confirm"}'
    sleep 0.5
    echo '{"cmd":"quit"}'
  ) | ./build/client.exe --client 127.0.0.1 1 --json 2>/dev/null | node tools/pretty_event.mjs "$label"
  echo
}

# -----------------------------------------------------------------------------
# KICH BAN 4 KHACH
# -----------------------------------------------------------------------------
one_customer \
  "KHACH 1 — Anh Nam" \
  "0901234567" \
  "Dan van phong, thuong sang goi Pho Bo + Tra Da. LFM hoc roi → top-3 se match thoi quen." \
  "P01 1;D01 1"

one_customer \
  "KHACH 2 — Chi Lan" \
  "0912345678" \
  "Sinh vien, thuong dat Com Tam + Nuoc Ngot. Top-3 phai khac hoan toan khach truoc." \
  "C01 1;D02 1"

one_customer \
  "KHACH 3 — SDT moi 0911111111" \
  "0911111111" \
  "Khach moi chua co lich su → cold-start. Goi y dua tren global pattern." \
  "P02 1;T01 1"

one_customer \
  "KHACH 4 — Chi Doan (SDT moi 0938888888), don LON" \
  "0938888888" \
  "Dat 5 mon lon → tong >= 2.000.000d → tu dong giam 25%." \
  "A01 20;C01 5;G01 10;P01 3;D01 5"

# -----------------------------------------------------------------------------
# Dong ca
# -----------------------------------------------------------------------------
section "SERVER: Thu ngan dong ca → xuat bao cao"
wait $SPID 2>/dev/null || true
tail -10 /tmp/demo_server.log

# -----------------------------------------------------------------------------
# Xem bao cao
# -----------------------------------------------------------------------------
section "BAO CAO CUOI CA (data/reports/report_$(date +%Y-%m-%d).txt)"
cat data/reports/report_$(date +%Y-%m-%d).txt

section "DEMO HOAN TAT"
echo "  Tat ca 4 don da duoc ghi nhan. LFM cap nhat online sau moi don."
echo "  Seed data preserved → chay demo lai cho ket qua tuong tu."
