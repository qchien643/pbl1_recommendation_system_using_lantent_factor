// dump_data.mjs — Doc cac file binary .dat va in ra text de dev inspect.
// Chay (tu goc project):    node tools/dump_data.mjs
//     in ra stdout — pipe vao file: node tools/dump_data.mjs > data/snapshot.txt
//
// Hoat dong voi state RUNTIME hien tai (users.dat + transactions.dat), bao gom
// cac user/don duoc them sau luc seed.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATA = path.join(ROOT, 'data');

const NAME_LEN = 40;
const DESC_LEN = 80;
const MAX_ITEMS = 5;

function readCString(buf, offset, maxLen) {
  const slice = buf.slice(offset, offset + maxLen);
  const z = slice.indexOf(0);
  return slice.toString('utf8', 0, z === -1 ? maxLen : z);
}

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}

// menu.txt: CODE|NAME|PRICE|CATEGORY
function loadMenu() {
  const map = {};
  const raw = fs.readFileSync(path.join(DATA, 'menu.txt'), 'utf8');
  raw.split('\n').forEach((line) => {
    line = line.trim();
    if (!line || line.startsWith('#')) return;
    const parts = line.split('|');
    if (parts.length >= 2) map[parts[0]] = parts[1];
  });
  return map;
}

// users.dat format:
//   [int userCount]
//   [char[11]  userPhone[N]]
//   [char[40]  userName[N]]
//   [char[80]  userDesc[N]]
//   [int       userTotalOrders[N]]
function loadUsers() {
  const buf = fs.readFileSync(path.join(DATA, 'users.dat'));
  const n = buf.readInt32LE(0);
  let off = 4;
  const users = [];
  for (let i = 0; i < n; i++) users.push({ phone: readCString(buf, off + i * 11, 10) });
  off += n * 11;
  for (let i = 0; i < n; i++) users[i].name = readCString(buf, off + i * NAME_LEN, NAME_LEN);
  off += n * NAME_LEN;
  for (let i = 0; i < n; i++) users[i].desc = readCString(buf, off + i * DESC_LEN, DESC_LEN);
  off += n * DESC_LEN;
  for (let i = 0; i < n; i++) users[i].totalOrders = buf.readInt32LE(off + i * 4);
  return users;
}

// transactions.dat format:
//   [int txnCount]
//   [int    userIdx[N]]
//   [char[20]  time[N]]
//   [char[10]  sess[N]]
//   [int    itemCount[N]]
//   [char[4]   itemCode[N][5]]
//   [int    itemQty[N][5]]
//   [float  subtotal[N]]
//   [float  discount[N]]
//   [float  total[N]]
function loadTransactions() {
  const buf = fs.readFileSync(path.join(DATA, 'transactions.dat'));
  const n = buf.readInt32LE(0);
  let off = 4;
  const txns = [];
  for (let i = 0; i < n; i++) txns.push({ userIdx: buf.readInt32LE(off + i * 4) });
  off += n * 4;
  for (let i = 0; i < n; i++) txns[i].time = readCString(buf, off + i * 20, 20);
  off += n * 20;
  for (let i = 0; i < n; i++) txns[i].sess = readCString(buf, off + i * 10, 10);
  off += n * 10;
  for (let i = 0; i < n; i++) txns[i].itemCount = buf.readInt32LE(off + i * 4);
  off += n * 4;
  for (let i = 0; i < n; i++) {
    const codes = [];
    for (let k = 0; k < MAX_ITEMS; k++) codes.push(readCString(buf, off + (i * MAX_ITEMS + k) * 4, 4));
    txns[i].itemCodes = codes;
  }
  off += n * MAX_ITEMS * 4;
  for (let i = 0; i < n; i++) {
    const qtys = [];
    for (let k = 0; k < MAX_ITEMS; k++) qtys.push(buf.readInt32LE(off + (i * MAX_ITEMS + k) * 4));
    txns[i].itemQtys = qtys;
  }
  off += n * MAX_ITEMS * 4;
  for (let i = 0; i < n; i++) txns[i].subtotal = buf.readFloatLE(off + i * 4);
  off += n * 4;
  for (let i = 0; i < n; i++) txns[i].discount = buf.readFloatLE(off + i * 4);
  off += n * 4;
  for (let i = 0; i < n; i++) txns[i].total    = buf.readFloatLE(off + i * 4);
  return txns;
}

// ----- main -----
const menu  = loadMenu();
const users = loadUsers();
let txns;
try { txns = loadTransactions(); } catch { txns = []; }

console.log('# SNAPSHOT DU LIEU RUNTIME');
console.log(`# Sinh boi tools/dump_data.mjs luc ${new Date().toLocaleString()}`);
console.log(`# Nguon: data/users.dat (${users.length} users) + data/transactions.dat (${txns.length} txns)`);
console.log('# ==========================================================\n');

users.forEach((u, uidx) => {
  const myTxns = txns.filter((t) => t.userIdx === uidx).sort((a, b) => a.time.localeCompare(b.time));
  const totalSpent = myTxns.reduce((s, t) => s + t.total, 0);
  const nameStr = u.name || '(chua dat ten)';
  console.log(`${u.phone} | ${nameStr.padEnd(20)} | ${String(u.totalOrders).padStart(2)} don | chi ${money(totalSpent).padStart(12)} | ${u.desc || ''}`);

  if (myTxns.length === 0) {
    console.log('   (chua co don nao)');
  } else {
    myTxns.forEach((t, i) => {
      const items = [];
      for (let k = 0; k < t.itemCount; k++) {
        const code = t.itemCodes[k];
        const name = menu[code] || code;
        items.push(`${code} ${name} x${t.itemQtys[k]}`);
      }
      const discStr = t.discount > 0 ? ` (-${money(t.discount)})` : '';
      const sessStr = t.sess.startsWith('SEED') ? '' : ` [${t.sess}]`;
      console.log(`   Don ${String(i + 1).padStart(2)} (${t.time})${sessStr}`);
      console.log(`     ${items.join(', ')} = ${money(t.total)}${discStr}`);
    });
  }
  console.log('');
});

// Tong ket
const totalTxn = txns.length;
const totalRev = txns.reduce((s, t) => s + t.total, 0);
console.log('# ==========================================================');
console.log(`# Tong users:        ${users.length}`);
console.log(`# Tong giao dich:    ${totalTxn}`);
console.log(`# Tong doanh thu:    ${money(totalRev)}`);
console.log('# ==========================================================');
