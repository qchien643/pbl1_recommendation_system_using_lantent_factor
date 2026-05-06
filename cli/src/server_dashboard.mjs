// server_dashboard.mjs — blessed-contrib dashboard cho server
// Chay: cd cli && npm run server

import fs from 'fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import blessed from 'blessed';
import contrib from 'blessed-contrib';
import { createServerIpc } from './server_ipc.js';
import { loadUsersTbl, loadTransactionsTbl, loadTxnItemsTbl, indexItemsByTxn } from './tbl_reader.mjs';

// Project root la 2 cap tren src/server_dashboard.mjs (chay tu bat ky CWD nao)
const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '../..');
const DATA_DIR   = path.join(PROJECT_ROOT, 'data');
const pathUsers     = path.join(DATA_DIR, 'users.tbl');         // mini-DBMS format
const pathTxns      = path.join(DATA_DIR, 'transactions.tbl');
const pathTxnItems  = path.join(DATA_DIR, 'transaction_items.tbl');
const pathMenu      = path.join(DATA_DIR, 'menu.txt');

const MAX_CLIENTS = 20;

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}
function nowStr() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

// --- State ---
let ready = false;
let port = 8888;
let sessionOpen = false;
let sessionCode = '';
let sessionStart = '';
let codeInput = '';
const stats = {
  totalOrdersToday: 0, revenueToday: 0, discountToday: 0, discountedOrders: 0,
  clientsConnected: 0, usersKnown: 0,
  totalSuggest: 0, suggestAccepted: 0
};

// --- Customer view state ---
let customerViewActive = false;
let _custData = {};
let _custPhones = [];
let _custSelectedIdx = 0;
let _menuMap = {}; // code -> name (tu menu.txt)

// --- blessed screen + grid ---
const screen = blessed.screen({ smartCSR: true, title: 'Restaurant Server Dashboard', fullUnicode: true });
const grid = new contrib.grid({ rows: 12, cols: 12, screen });

// Row 0-2: Title + Session
const headerBox = grid.set(0, 0, 3, 7, blessed.box, {
  label: ' Viet Phong Server ',
  border: { type: 'line' },
  style: { border: { fg: 'cyan' }, label: { fg: 'cyan', bold: true } },
  tags: true,
  padding: { left: 1, right: 1 }
});
const sessionBox = grid.set(0, 7, 3, 5, blessed.box, {
  label: ' Session ',
  border: { type: 'line' },
  style: { border: { fg: 'yellow' }, label: { fg: 'yellow', bold: true } },
  tags: true,
  padding: { left: 1, right: 1 }
});

// Row 3-5: Live Stats + Top Sellers + Top Customers (replaces LCD gauges)
const liveStatsBox = grid.set(3, 0, 3, 4, blessed.box, {
  label: ' ▣ Live Stats ',
  border: { type: 'line' },
  style: { border: { fg: 'cyan' }, label: { fg: 'cyan', bold: true } },
  tags: true,
  padding: { left: 1, right: 1 }
});
const topSellersBox = grid.set(3, 4, 3, 4, blessed.box, {
  label: ' ⚑ Top Sellers Today ',
  border: { type: 'line' },
  style: { border: { fg: 'yellow' }, label: { fg: 'yellow', bold: true } },
  tags: true,
  padding: { left: 1, right: 1 }
});
const topCustomersBox = grid.set(3, 8, 3, 4, blessed.box, {
  label: ' ✦ Top Customers ',
  border: { type: 'line' },
  style: { border: { fg: 'green' }, label: { fg: 'green', bold: true } },
  tags: true,
  padding: { left: 1, right: 1 }
});

// Row 6-11 (6 rows — 50% chieu cao grid): Activity Log (default) HOAC Customer panel (toggled bang Tab).
// Hai view cung share vi tri, show/hide dan xen.
const activityLog = grid.set(6, 0, 6, 12, contrib.log, {
  label: ' Activity Log  [Tab → Customers] ',
  border: { type: 'line' },
  style: { border: { fg: 'white' }, label: { fg: 'white', bold: true } },
  bufferLength: 100,
  tags: true
});

// Customer panel tai cung vi tri Activity Log (hidden ban dau)
const custTable = grid.set(6, 0, 6, 6, contrib.table, {
  label: ' Customers  [↑↓ select]  [Enter view]  [Tab → Log] ',
  columnSpacing: 1,
  columnWidth: [13, 17, 4],
  keys: true,
  vi: true,
  fg: 'white',
  selectedFg: 'black',
  selectedBg: 'green',
  border: { type: 'line' },
  style: { border: { fg: 'green' }, label: { fg: 'green', bold: true }, header: { fg: 'yellow', bold: true } }
});

const custDetail = grid.set(6, 6, 6, 6, contrib.log, {
  label: ' Transaction History ',
  border: { type: 'line' },
  style: { border: { fg: 'cyan' }, label: { fg: 'cyan', bold: true } },
  bufferLength: 500,
  tags: true,
  keys: true,
  vi: true,
  scrollable: true,
  alwaysScroll: true,
  scrollbar: { ch: ' ', style: { bg: 'cyan' } },
  mouse: true
});

// Hide customer panel initially
custTable.hide();
custDetail.hide();

// --- Helpers ---
function renderHeader() {
  // Big block text "VIET PHONG" (cfonts 'chrome' font, hardcoded để render ổn định
  // không phụ thuộc terminal width detection của cfonts). Box-drawing chars sạch nét.
  const lines = [
    '{bold}{cyan-fg} ╦  ╦ ╦ ╔═╗ ╔╦╗      ╔═╗ ╦ ╦ ╔═╗ ╔╗╔ ╔═╗{/}',
    '{bold}{cyan-fg} ╚╗╔╝ ║ ║╣   ║       ╠═╝ ╠═╣ ║ ║ ║║║ ║ ╦{/}',
    '{bold}{cyan-fg}  ╚╝  ╩ ╚═╝  ╩       ╩   ╩ ╩ ╚═╝ ╝╚╝ ╚═╝{/}',
    '{yellow-fg} (=^.^=){/}  {magenta-fg}F&B Smart Order{/}  {gray-fg}·{/}  {green-fg}LFM Recommendations{/}',
    '{gray-fg}                              DUT PBL1 — De 702{/}'
  ];
  headerBox.setContent(lines.join('\n'));
}
function renderSession() {
  let lines;
  if (sessionOpen) {
    lines = [
      '{green-fg}{bold}● SESSION OPEN{/}{/bold}',
      `Code    : {cyan-fg}${sessionCode}{/}`,
      `Started : {cyan-fg}${sessionStart}{/}`,
      '',
      `Re-enter code + Enter to CLOSE:`,
      `> {cyan-fg}${codeInput}{/}{gray-fg}_{/}`
    ];
  } else {
    lines = [
      '{yellow-fg}{bold}○ SESSION CLOSED{/}{/bold}',
      '',
      `Enter CODE (1-9 digits) + Enter:`,
      `> {cyan-fg}${codeInput}{/}{gray-fg}_{/}`,
      '',
      '{gray-fg}e.g., 1234{/}'
    ];
  }
  sessionBox.setContent(lines.join('\n'));
}
// --- Aggregate state cho 3 stats panel (rebuild tu .tbl moi khi co order) ---
let _aggLastTxnCount = -1;        // throttle: chi rebuild khi txnCount tang
let _aggTopSellers = [];           // [{code, name, qty, revenue}]
let _aggTopCustomers = [];         // [{userId, phone, name, count, total}]
let _aggSessionStats = {           // tinh toan tu txns co session_code = current
  count: 0, revenue: 0, discountTotal: 0, discountedCount: 0, uniqueUsers: 0
};

function rebuildAggregates() {
  try {
    const txns = loadTransactionsTbl(pathTxns);
    const items = loadTxnItemsTbl(pathTxnItems);
    const users = loadUsersTbl(pathUsers);
    if (txns.length === _aggLastTxnCount && _aggLastTxnCount > 0) return;  // no change
    _aggLastTxnCount = txns.length;

    const userMap = new Map(users.map(u => [u.userId, u]));

    // Group by item_code (tan dung BTree(item_code) capability)
    const itemAgg = new Map();
    for (const it of items) {
      const cur = itemAgg.get(it.code) ?? { qty: 0, revenue: 0 };
      cur.qty += it.qty;
      cur.revenue += it.qty * it.price;
      itemAgg.set(it.code, cur);
    }
    _aggTopSellers = [...itemAgg.entries()]
      .sort((a, b) => b[1].qty - a[1].qty)
      .slice(0, 5)
      .map(([code, c]) => ({ code, name: _menuMap[code] || code, qty: c.qty, revenue: c.revenue }));

    // Group by user_id (tan dung BTree(user_id) capability)
    const userAgg = new Map();
    for (const t of txns) {
      const s = userAgg.get(t.userId) ?? { count: 0, total: 0 };
      s.count++;
      s.total += t.total;
      userAgg.set(t.userId, s);
    }
    _aggTopCustomers = [...userAgg.entries()]
      .sort((a, b) => b[1].total - a[1].total)
      .slice(0, 5)
      .map(([uid, s]) => {
        const u = userMap.get(uid);
        return {
          userId: uid,
          phone: u?.phone || `uid${uid}`,
          name:  u?.name || '(unnamed)',
          count: s.count,
          total: s.total
        };
      });

    // Session stats (BTree(session_code) - filter theo current session)
    if (sessionOpen && sessionCode) {
      const sessTxns = txns.filter(t => t.sess === sessionCode);
      _aggSessionStats.count = sessTxns.length;
      _aggSessionStats.revenue = sessTxns.reduce((a, t) => a + t.total, 0);
      _aggSessionStats.discountTotal = sessTxns.reduce((a, t) => a + t.discount, 0);
      _aggSessionStats.discountedCount = sessTxns.filter(t => t.discount > 0).length;
      _aggSessionStats.uniqueUsers = new Set(sessTxns.map(t => t.userId)).size;
    } else {
      _aggSessionStats = { count: 0, revenue: 0, discountTotal: 0, discountedCount: 0, uniqueUsers: 0 };
    }
  } catch (e) { /* file chua co — bo qua */ }
}

function renderLiveStats() {
  const s = _aggSessionStats;
  const acc = stats.totalSuggest > 0
    ? Math.round(100 * stats.suggestAccepted / stats.totalSuggest) : 0;
  const avg = s.count > 0 ? Math.round(s.revenue / s.count) : 0;
  const lines = [
    `{gray-fg}Session:{/}        {bold}${sessionOpen ? '{green-fg}● ' + sessionCode + '{/}' : '{yellow-fg}○ closed{/}'}{/bold}`,
    `{gray-fg}Orders today:{/}   {cyan-fg}${s.count}{/}`,
    `{gray-fg}Revenue today:{/}  {bold}{green-fg}${money(s.revenue)}{/}{/bold}`,
    `{gray-fg}Discounts:{/}      {yellow-fg}${s.discountedCount}{/} {gray-fg}(-${money(s.discountTotal)}){/}`,
    `{gray-fg}Avg ticket:{/}     {magenta-fg}${money(avg)}{/}`,
    `{gray-fg}Unique guests:{/}  {cyan-fg}${s.uniqueUsers}{/}`,
    `{gray-fg}Active clients:{/} {cyan-fg}${stats.clientsConnected}{/}/${MAX_CLIENTS}  {gray-fg}LFM ${acc}%{/}`
  ];
  liveStatsBox.setContent(lines.join('\n'));
}

function renderTopSellers() {
  if (_aggTopSellers.length === 0) {
    topSellersBox.setContent('{gray-fg}(chưa có dữ liệu){/}');
    return;
  }
  const medals = ['{yellow-fg}①{/}', '{yellow-fg}②{/}', '{yellow-fg}③{/}', ' ④', ' ⑤'];
  const lines = _aggTopSellers.map((s, i) => {
    const name = (s.name || '').slice(0, 14);
    return `${medals[i]} {bold}${s.code}{/} ${name.padEnd(14)} {cyan-fg}x${String(s.qty).padStart(3)}{/} {green-fg}${money(s.revenue).padStart(10)}{/}`;
  });
  topSellersBox.setContent(lines.join('\n'));
}

function renderTopCustomers() {
  if (_aggTopCustomers.length === 0) {
    topCustomersBox.setContent('{gray-fg}(chưa có dữ liệu){/}');
    return;
  }
  const medals = ['{yellow-fg}①{/}', '{yellow-fg}②{/}', '{yellow-fg}③{/}', ' ④', ' ⑤'];
  const lines = _aggTopCustomers.map((c, i) => {
    const name = (c.name || '(unnamed)').slice(0, 14);
    return `${medals[i]} ${c.phone} ${name.padEnd(14)} {cyan-fg}x${String(c.count).padStart(2)}{/} {green-fg}${money(c.total).padStart(10)}{/}`;
  });
  topCustomersBox.setContent(lines.join('\n'));
}

function redraw() {
  renderHeader();
  renderSession();
  renderLiveStats();
  renderTopSellers();
  renderTopCustomers();
  screen.render();
}

// --- Customer panel functions ---

// Doc menu.txt de map code -> name
function loadMenu() {
  _menuMap = {};
  try {
    const raw = fs.readFileSync(pathMenu, 'utf8');
    raw.split('\n').forEach((line) => {
      line = line.trim();
      if (!line || line.startsWith('#')) return;
      const parts = line.split('|');
      if (parts.length < 2) return;
      _menuMap[parts[0]] = parts[1];
    });
  } catch {}
}

// Doc cstring tu buffer (tu offset, max len): dung indexOf null-terminator.
function readCString(buf, offset, maxLen) {
  const slice = buf.slice(offset, offset + maxLen);
  const nullIdx = slice.indexOf(0);
  return slice.toString('utf8', 0, nullIdx === -1 ? maxLen : nullIdx);
}

// Doc binary data/users.dat (format moi):
//   [int userCount]
//   [char[11]  userPhone[N]]
//   [char[40]  userName[N]]
//   [char[80]  userDesc[N]]
//   [int       userTotalOrders[N]]
function loadUsersDat() {
  const users = [];
  try {
    const buf = fs.readFileSync(pathUsers);
    if (buf.length < 4) return users;
    const NAME_LEN = 40, DESC_LEN = 80;
    const n = buf.readInt32LE(0);
    if (n <= 0 || n > 1000) return users;
    let off = 4;
    const phones = [], names = [], descs = [];
    for (let i = 0; i < n; i++) phones.push(readCString(buf, off + i * 11, 10));
    off += n * 11;
    for (let i = 0; i < n; i++) names.push(readCString(buf, off + i * NAME_LEN, NAME_LEN));
    off += n * NAME_LEN;
    for (let i = 0; i < n; i++) descs.push(readCString(buf, off + i * DESC_LEN, DESC_LEN));
    off += n * DESC_LEN;
    for (let i = 0; i < n; i++) {
      users.push({
        phone: phones[i],
        name: names[i],
        desc: descs[i],
        totalOrders: buf.readInt32LE(off + i * 4)
      });
    }
  } catch {}
  return users;
}

// Doc binary data/transactions.dat:
//   [int txnCount]
//   [int    txnUserIdx[N]]
//   [char[20]  txnTime[N]]
//   [char[10]  txnSessionCode[N]]
//   [int    txnItemCount[N]]
//   [char[4]   txnItemCode[N][5]]
//   [int    txnItemQty[N][5]]
//   [float  txnSubtotal[N]]
//   [float  txnDiscount[N]]
//   [float  txnTotal[N]]
function loadTransactionsDat() {
  const txns = [];
  try {
    const buf = fs.readFileSync(pathTxns);
    if (buf.length < 4) return txns;
    const MAX_ITEMS = 5;
    const n = buf.readInt32LE(0);
    if (n <= 0 || n > 5000) return txns;
    let off = 4;
    const userIdx = [], time = [], sess = [], itemCount = [];
    const itemCode = [], itemQty = [];
    const subtotal = [], discount = [], total = [];
    for (let i = 0; i < n; i++) userIdx.push(buf.readInt32LE(off + i * 4));
    off += n * 4;
    for (let i = 0; i < n; i++) time.push(readCString(buf, off + i * 20, 20));
    off += n * 20;
    for (let i = 0; i < n; i++) sess.push(readCString(buf, off + i * 10, 10));
    off += n * 10;
    for (let i = 0; i < n; i++) itemCount.push(buf.readInt32LE(off + i * 4));
    off += n * 4;
    for (let i = 0; i < n; i++) {
      const codes = [];
      for (let k = 0; k < MAX_ITEMS; k++) {
        codes.push(readCString(buf, off + (i * MAX_ITEMS + k) * 4, 4));
      }
      itemCode.push(codes);
    }
    off += n * MAX_ITEMS * 4;
    for (let i = 0; i < n; i++) {
      const qtys = [];
      for (let k = 0; k < MAX_ITEMS; k++) {
        qtys.push(buf.readInt32LE(off + (i * MAX_ITEMS + k) * 4));
      }
      itemQty.push(qtys);
    }
    off += n * MAX_ITEMS * 4;
    for (let i = 0; i < n; i++) subtotal.push(buf.readFloatLE(off + i * 4));
    off += n * 4;
    for (let i = 0; i < n; i++) discount.push(buf.readFloatLE(off + i * 4));
    off += n * 4;
    for (let i = 0; i < n; i++) total.push(buf.readFloatLE(off + i * 4));

    for (let i = 0; i < n; i++) {
      const items = [];
      for (let k = 0; k < itemCount[i]; k++) {
        items.push({ code: itemCode[i][k], qty: itemQty[i][k] });
      }
      txns.push({
        userIdx: userIdx[i],
        time: time[i],
        session: sess[i],
        items,
        subtotal: subtotal[i],
        discount: discount[i],
        total: total[i]
      });
    }
  } catch {}
  return txns;
}

// Load all customer data từ mini-DBMS .tbl files
//   users.tbl              — 1 row per khách
//   transactions.tbl       — 1 row per đơn (header)
//   transaction_items.tbl  — N rows per đơn (line items)
function loadCustomerData() {
  _custData = {};
  loadMenu();

  // 1. Load danh sách khách
  const users = loadUsersTbl(pathUsers);
  // Map userId → phone (user_id là cột rõ ràng trong table mới)
  const phoneByUserId = new Map();
  users.forEach((u) => {
    phoneByUserId.set(u.userId, u.phone);
    _custData[u.phone] = {
      userId: u.userId,
      name: u.name || '',
      desc: u.desc || '',
      count: u.totalOrders,
      totalSpent: 0,
      txns: []
    };
  });

  // 2. Load transactions + items, group items theo txn_id
  const txns = loadTransactionsTbl(pathTxns);
  const items = loadTxnItemsTbl(pathTxnItems);
  const itemsByTxn = indexItemsByTxn(items);

  txns.forEach((t) => {
    const phone = phoneByUserId.get(t.userId);
    if (!phone || !_custData[phone]) return;
    _custData[phone].totalSpent += t.total;
    const txItems = itemsByTxn.get(t.txnId) || [];
    _custData[phone].txns.push({
      ts: t.ts,
      sess: t.sess,
      items: txItems.map((it) => ({ code: it.code, qty: it.qty })),
      total: t.total,
      discount: t.discount
    });
  });

  _custPhones = Object.keys(_custData).sort();
}

function renderCustTable() {
  const rows = _custPhones.map((p) => {
    const d = _custData[p];
    const shortName = (d.name || '(unnamed)').slice(0, 16);
    return [p, shortName, String(d.count)];
  });
  custTable.setData({
    headers: ['Phone', 'Name', 'Orders'],
    data: rows.length > 0 ? rows : [['(no data)', '—', '—']]
  });
}

function showCustDetail(phone) {
  custDetail.logLines = [];
  custDetail.setContent('');
  if (!phone || !_custData[phone]) {
    custDetail.log('{gray-fg}◇ Select a customer to view details.{/}');
    return;
  }
  const d = _custData[phone];

  // Header: Name + Phone + Description
  const nameStr = d.name || '{gray-fg}(unnamed){/}';
  custDetail.log(`{cyan-fg}◆ ${nameStr}  ·  ${phone}{/}`);
  if (d.desc) custDetail.log(`  {gray-fg}${d.desc}{/}`);
  custDetail.log(`{green-fg}▲ ${d.count} orders  ·  Spent: ${money(d.totalSpent)}{/}`);
  custDetail.log('{yellow-fg}' + '─'.repeat(44) + '{/}');

  // All transactions (no cap)
  if (d.txns.length > 0) {
    custDetail.log(`{magenta-fg}★ History: ${d.txns.length} orders (↑↓/PgUp/PgDn to scroll):{/}`);
    // Newest first
    const sorted = [...d.txns].sort((a, b) => (b.ts || '').localeCompare(a.ts || ''));
    sorted.forEach((t, i) => {
      const itemsStr = t.items.map((it) => {
        const name = _menuMap[it.code] || it.code;
        return `{yellow-fg}${it.code}{/} ${name} {cyan-fg}x${it.qty}{/}`;
      }).join(', ');
      const discStr = t.discount > 0 ? ` {yellow-fg}(-${money(t.discount)}){/}` : '';
      // Hide SEED session codes (demo data)
      const sessStr = (t.sess && !t.sess.startsWith('SEED')) ? ` {magenta-fg}[${t.sess}]{/}` : '';
      custDetail.log(`  {white-fg}◉ Order #${i + 1} · ${t.ts}{/}${sessStr}`);
      custDetail.log(`    ${itemsStr}`);
      custDetail.log(`    = {green-fg}${money(t.total)}${discStr}{/}`);
    });
  } else {
    custDetail.log('{gray-fg}(no orders yet){/}');
  }
  if (custDetail.setScroll) custDetail.setScroll(0);
}

function enterCustomerView() {
  loadCustomerData();
  renderCustTable();
  _custSelectedIdx = 0;
  if (_custPhones.length > 0) {
    custTable.rows.select(0); // list item 0 = khach #1 (header o parent Box content)
    showCustDetail(_custPhones[0]);
  } else {
    showCustDetail(null);
  }
  activityLog.hide();
  custTable.show();
  custDetail.show();
  custTable.focus();
  screen.render();
}

function enterActivityView() {
  custTable.hide();
  custDetail.hide();
  activityLog.show();
  screen.render();
}

// Wire custTable row selection to update detail panel (highlight change via arrow keys)
// Note: blessed-contrib table dat header vao parent Box content, KHONG phai item[0] cua list.
// Vi vay list index 0 = khach #1, list index N = khach #N+1 (khong can -1).
custTable.rows.on('select item', (item, index) => {
  if (index >= 0 && index < _custPhones.length) {
    _custSelectedIdx = index;
    showCustDetail(_custPhones[index]);
    screen.render();
  }
});

// Enter on custTable -> move focus to custDetail so user can scroll history
custTable.rows.on('select', () => {
  custDetail.focus();
  custDetail.setLabel(' Transaction History  [Esc/← back] ');
  custTable.setLabel(' Customers ');
  screen.render();
});

// Scroll + navigation cho custDetail: dung on('keypress') de CHI bat khi focused
// (element.key() la global qua program — conflict voi custTable's arrow keys)
custDetail.on('keypress', (ch, key) => {
  if (screen.focused !== custDetail) return;
  if (!key) return;
  const name = key.name;
  const h = custDetail.height || 5;
  if (name === 'up' || name === 'k') {
    custDetail.scroll(-1);
    screen.render();
  } else if (name === 'down' || name === 'j') {
    custDetail.scroll(1);
    screen.render();
  } else if (name === 'pageup') {
    custDetail.scroll(-(h - 1));
    screen.render();
  } else if (name === 'pagedown') {
    custDetail.scroll(h - 1);
    screen.render();
  } else if (name === 'home' || (name === 'g' && !key.shift)) {
    custDetail.setScroll(0);
    screen.render();
  } else if (name === 'end' || (name === 'g' && key.shift)) {
    custDetail.setScrollPerc(100);
    screen.render();
  } else if (name === 'escape' || name === 'left') {
    custTable.focus();
    custTable.setLabel(' Customers  [↑↓ select]  [Enter view] ');
    custDetail.setLabel(' Transaction History ');
    screen.render();
  }
});

// Re-draw mỗi giây để timestamp header + HB-ago (nếu cần sau này) được fresh
setInterval(() => { redraw(); }, 1000);

// --- Wire IPC ---
const ipc = createServerIpc();

// server emit "server_started" sau Spring refactor (truoc la "ready")
ipc.on('server_started', (e) => {
  ready = true; port = e.port;
  loadMenu();
  activityLog.log(`{green-fg}${nowStr()} ━ Server READY{/}  {gray-fg}port ${e.port}{/}`);
  redraw();
});
ipc.on('menu_loaded', (e) => {
  activityLog.log(`{gray-fg}${nowStr()} • Menu loaded · ${e.count} items{/}`);
  redraw();
});
ipc.on('session_opened', (e) => {
  sessionOpen = true; sessionCode = e.code; sessionStart = e.dateTime; codeInput = '';
  rebuildAggregates();   // load aggregates tu .tbl files (vd seed data)
  activityLog.log(`{bold}{blue-fg}${nowStr()} ━━ SESSION OPENED  ·  code ${e.code}{/}{/bold}`);
  redraw();
});
ipc.on('session_closed', (e) => {
  sessionOpen = false;
  activityLog.log(`{yellow-fg}${nowStr()} Session CLOSED · ${e.totalOrders} orders · revenue ${money(e.totalRevenue)}{/}`);
  redraw();
  setTimeout(() => { ipc.quit(); process.exit(0); }, 2500);
});
// server emit "client_connect"/"client_disconnect" sau Spring refactor
ipc.on('client_connect', (e) => {
  const tbl = String(e.slot + 1).padStart(2, '0');
  stats.clientsConnected++;
  activityLog.log(`{cyan-fg}${nowStr()} ▼ Table ${tbl}{/}  {gray-fg}connected{/}`);
  redraw();
});
ipc.on('client_disconnect', (e) => {
  const tbl = String(e.slot + 1).padStart(2, '0');
  if (stats.clientsConnected > 0) stats.clientsConnected--;
  activityLog.log(`{gray-fg}${nowStr()} ▲ Table ${tbl}  disconnected{/}`);
  redraw();
});
ipc.on('user_login', (e) => {
  const tbl = String(e.slot + 1).padStart(2, '0');
  const tag = e.isNew ? '{green-fg}NEW{/}' : `{gray-fg}${e.orderCount} prior orders{/}`;
  activityLog.log(`{blue-fg}${nowStr()} ▼ LOGIN{/}   {white-fg}Table ${tbl}{/}  ·  ${e.phone}  ·  ${tag}`);
  stats.totalSuggest++;
  redraw();
});
ipc.on('user_register', (e) => {
  const tbl = String(e.slot + 1).padStart(2, '0');
  activityLog.log(`{magenta-fg}${nowStr()} + REGISTER{/} {white-fg}Table ${tbl}{/}  ·  ${e.phone}  ·  {bold}${e.name}{/}  {green-fg}(NEW){/}`);
  redraw();
});
ipc.on('item_added', (e) => {
  const tbl = String(e.slot + 1).padStart(2, '0');
  activityLog.log(`{gray-fg}${nowStr()} • Table ${tbl}  added {yellow-fg}${e.code}{/}{/}`);
  stats.totalSuggest++;
  redraw();
});

// Helper: render rich one-line entry cho ORDER_SUBMIT (re-load .tbl de lay items + name)
function renderOrderEntry(e) {
  let phone = '?', name = '?', itemsStr = `${e.items} items`;
  try {
    const txns = loadTransactionsTbl(pathTxns);
    const items = indexItemsByTxn(loadTxnItemsTbl(pathTxnItems));
    const users = loadUsersTbl(pathUsers);
    // orderId = txnId + 1 (theo ORDER_ACK format trong order_controller.cpp)
    const txnId = e.orderId - 1;
    const t = txns.find(x => x.txnId === txnId);
    if (t) {
      const u = users.find(x => x.userId === t.userId);
      if (u) { phone = u.phone; name = u.name || '(unnamed)'; }
      const its = items.get(txnId) || [];
      itemsStr = its.map(it => `${it.code}x${it.qty}`).join(' + ');
    }
  } catch {}
  const tbl = String(e.slot + 1).padStart(2, '0');
  const isBig = e.discount > 0;
  if (isBig) {
    return `{yellow-fg}{bold}${nowStr()} ★ ORDER #${String(e.orderId).padStart(3,'0')}{/}{/bold}  ·  Table ${tbl}  ·  ${phone} ${name}  ·  ${itemsStr}  ·  {bold}${money(e.total)}{/}  {magenta-fg}(-25% = ${money(e.discount)}){/}`;
  }
  return `{green-fg}${nowStr()} ✓ ORDER #${String(e.orderId).padStart(3,'0')}{/}  ·  Table ${tbl}  ·  ${phone} ${name}  ·  ${itemsStr}  ·  {bold}${money(e.total)}{/}`;
}

ipc.on('order_submitted', (e) => {
  // Stats counters live (incremental, song song voi aggregate tu .tbl)
  stats.totalOrdersToday++;
  stats.revenueToday += e.total;
  if (e.discount > 0) {
    stats.discountToday += e.discount;
    stats.discountedOrders++;
  }
  // Refresh aggregates tu .tbl files (BTree group-by capability)
  rebuildAggregates();
  // Activity Stream entry rich format
  activityLog.log(renderOrderEntry(e));
  redraw();
});
// 'stats' event: legacy, sau refactor server khong emit nua. Stats compute local + tu .tbl.
ipc.on('suggest', (e) => {
  // Server gui suggest top-3 → tang counter de track LFM accept rate sau nay
  if (e.items && e.items.length > 0) stats.suggestAccepted++;
  redraw();
});
// heartbeat event: da bo phan hien thi — nhan im lang de khong log spam
ipc.on('heartbeat', () => {});
ipc.on('error', (e) => {
  activityLog.log(`{red-fg}${nowStr()} ERROR: ${e.message || 'unknown'}{/}`);
  redraw();
});
ipc.onExit(() => {
  activityLog.log(`{red-fg}${nowStr()} server.exe exited{/}`);
  redraw();
});

// Initial hint in Activity Log
activityLog.log(`{gray-fg}${nowStr()} Enter CODE + Enter to open session · Esc/Ctrl+C to quit{/}`);

// --- Key bindings ---
screen.key(['escape', 'C-c'], () => { ipc.quit(); process.exit(0); });
screen.key(['tab'], () => {
  // Gate: chi cho xem lich su khach hang khi CA DA MO.
  // Muc dich: trong lung thoi ky giua 2 ca khong lo du lieu ra UI.
  if (!customerViewActive && !sessionOpen) {
    activityLog.log(`{red-fg}${nowStr()} Must open SESSION before viewing customer history{/}`);
    redraw();
    return;
  }
  customerViewActive = !customerViewActive;
  if (customerViewActive) enterCustomerView();
  else enterActivityView();
});
screen.key(['enter'], () => {
  if (customerViewActive) return;
  if (codeInput.length === 0) return;
  if (!sessionOpen) ipc.openSession(codeInput);
  else ipc.closeSession(codeInput);
});
screen.key(['backspace'], () => {
  if (customerViewActive) return;
  codeInput = codeInput.slice(0, -1);
  redraw();
});
for (let i = 0; i <= 9; i++) {
  screen.key([String(i)], () => {
    if (customerViewActive) return;
    if (codeInput.length < 9) codeInput += String(i);
    redraw();
  });
}

// Initial render
redraw();
