// tbl_reader.mjs — Reader cho file .tbl (mini-DBMS format mới).
//
// Format:
//   [16 bytes magic]   "PBL1DBv1\0\0\0\0\0\0\0\0"
//   [4 bytes uint32]   schema_hash (CRC32 - không kiểm)
//   [4 bytes uint32]   row_count
//   [4 bytes uint32]   row_size_bytes
//   [row_count × row_size_bytes]   fixed-width rows
//
// Schema khớp với shared/db/db_schema.cpp.

import fs from 'fs';

const MAGIC = Buffer.from('PBL1DBv1\0\0\0\0\0\0\0\0', 'binary');
const HEADER_SIZE = 16 + 4 + 4 + 4;  // 28 bytes

// Đọc fixed-width string, trim ở first NUL
function readStr(buf, off, len) {
  let end = off;
  const limit = off + len;
  while (end < limit && buf[end] !== 0) end++;
  return buf.toString('utf8', off, end);
}

// Mở file .tbl, trả về { rowCount, rowSize, rowsBuf } hoặc null nếu không hợp lệ.
function openTbl(path) {
  if (!fs.existsSync(path)) return null;
  const buf = fs.readFileSync(path);
  if (buf.length < HEADER_SIZE) return null;
  if (!buf.slice(0, 16).equals(MAGIC)) return null;
  const rowCount = buf.readUInt32LE(20);
  const rowSize  = buf.readUInt32LE(24);
  const expected = HEADER_SIZE + rowCount * rowSize;
  if (buf.length < expected) return null;
  return { rowCount, rowSize, rowsBuf: buf.slice(HEADER_SIZE), buf };
}

// users.tbl row: user_id INT64 | phone STR(11) | name STR(40) | desc STR(80) | total_orders INT64 | created_at INT64
// row_size = 8+11+40+80+8+8 = 155
export function loadUsersTbl(path) {
  const t = openTbl(path);
  if (!t) return [];
  const out = [];
  for (let i = 0; i < t.rowCount; i++) {
    const off = i * t.rowSize;
    const row = t.rowsBuf;
    const userId      = Number(row.readBigInt64LE(off));
    const phone       = readStr(row, off + 8,  11);
    const name        = readStr(row, off + 19, 40);
    const desc        = readStr(row, off + 59, 80);
    const totalOrders = Number(row.readBigInt64LE(off + 139));
    const createdAt   = Number(row.readBigInt64LE(off + 147));
    out.push({ userId, phone, name, desc, totalOrders, createdAt });
  }
  return out;
}

// transactions.tbl row: txn_id INT64 | user_id INT64 | session_code STR(10) | ts STR(20) | subtotal F64 | discount F64 | total F64
// row_size = 8+8+10+20+8+8+8 = 70
export function loadTransactionsTbl(path) {
  const t = openTbl(path);
  if (!t) return [];
  const out = [];
  for (let i = 0; i < t.rowCount; i++) {
    const off = i * t.rowSize;
    const row = t.rowsBuf;
    const txnId    = Number(row.readBigInt64LE(off));
    const userId   = Number(row.readBigInt64LE(off + 8));
    const sess     = readStr(row, off + 16, 10);
    const ts       = readStr(row, off + 26, 20);
    const subtotal = row.readDoubleLE(off + 46);
    const discount = row.readDoubleLE(off + 54);
    const total    = row.readDoubleLE(off + 62);
    out.push({ txnId, userId, sess, ts, subtotal, discount, total });
  }
  return out;
}

// transaction_items.tbl row: txn_id INT64 | seq INT64 | item_code STR(4) | qty INT64 | price F64
// row_size = 8+8+4+8+8 = 36
export function loadTxnItemsTbl(path) {
  const t = openTbl(path);
  if (!t) return [];
  const out = [];
  for (let i = 0; i < t.rowCount; i++) {
    const off = i * t.rowSize;
    const row = t.rowsBuf;
    const txnId = Number(row.readBigInt64LE(off));
    const seq   = Number(row.readBigInt64LE(off + 8));
    const code  = readStr(row, off + 16, 4);
    const qty   = Number(row.readBigInt64LE(off + 20));
    const price = row.readDoubleLE(off + 28);
    out.push({ txnId, seq, code, qty, price });
  }
  return out;
}

// Helper: gom items theo txn_id → Map<txnId, [{code, qty, price}, ...]>
export function indexItemsByTxn(items) {
  const m = new Map();
  for (const it of items) {
    if (!m.has(it.txnId)) m.set(it.txnId, []);
    m.get(it.txnId).push({ code: it.code, qty: it.qty, price: it.price, seq: it.seq });
  }
  for (const arr of m.values()) arr.sort((a, b) => a.seq - b.seq);
  return m;
}
