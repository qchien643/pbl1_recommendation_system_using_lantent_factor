// server_dashboard.mjs — blessed-contrib dashboard cho server
// Chay: cd cli && npm run server

import fs from 'fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import blessed from 'blessed';
import contrib from 'blessed-contrib';
import { createServerIpc } from './server_ipc.js';

// Project root la 2 cap tren src/server_dashboard.mjs (chay tu bat ky CWD nao)
const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '../..');
const DATA_DIR   = path.join(PROJECT_ROOT, 'data');
const pathUsers  = path.join(DATA_DIR, 'users.dat');
const pathTxns   = path.join(DATA_DIR, 'transactions.dat');
const pathMenu   = path.join(DATA_DIR, 'menu.txt');

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

// Row 3-5: Gauges + Stats
const gaugeOrders = grid.set(3, 0, 3, 3, contrib.lcd, {
  label: ' Orders Today ',
  segmentWidth: 0.05,
  segmentInterval: 0.11,
  strokeWidth: 0.1,
  elements: 4,
  display: 0,
  elementSpacing: 4,
  elementPadding: 2,
  color: 'green',
  style: { border: { fg: 'green' }, label: { fg: 'green' } }
});
const gaugeRevenue = grid.set(3, 3, 3, 4, contrib.lcd, {
  label: ' Revenue (K VND) ',
  segmentWidth: 0.05,
  segmentInterval: 0.11,
  strokeWidth: 0.1,
  elements: 7,
  display: 0,
  elementSpacing: 3,
  elementPadding: 2,
  color: 'cyan',
  style: { border: { fg: 'cyan' }, label: { fg: 'cyan' } }
});
const miscStats = grid.set(3, 7, 3, 5, blessed.box, {
  label: ' Overall Stats ',
  border: { type: 'line' },
  style: { border: { fg: 'blue' }, label: { fg: 'blue', bold: true } },
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
  const status = ready ? '{green-fg}● READY{/}' : '{yellow-fg}○ starting...{/}';
  const lines = [
    '{bold}{cyan-fg}VIET PHONG RESTAURANT{/}{/bold}',
    '',
    `Server ${status}  ·  port ${port}`,
    `Time: ${new Date().toLocaleString()}`
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
function renderGauges() {
  gaugeOrders.setDisplay(stats.totalOrdersToday);
  const revK = Math.round(stats.revenueToday / 1000);
  gaugeRevenue.setDisplay(revK);
}
function renderMisc() {
  const acc = stats.totalSuggest > 0
    ? Math.round(100 * stats.suggestAccepted / stats.totalSuggest) : 0;
  const lines = [
    `Clients:     {cyan-fg}${stats.clientsConnected}{/}/${MAX_CLIENTS}`,
    `Users:       {cyan-fg}${stats.usersKnown}{/}`,
    `Discount:    {yellow-fg}${money(stats.discountToday)}{/}`,
    `Discounted:  {yellow-fg}${stats.discountedOrders}{/}`,
    `LFM sugg:    {magenta-fg}${stats.suggestAccepted}/${stats.totalSuggest}{/} ({magenta-fg}${acc}%{/})`
  ];
  miscStats.setContent(lines.join('\n'));
}
function redraw() {
  renderHeader();
  renderSession();
  renderGauges();
  renderMisc();
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

// Load all customer data: users.dat (source of truth for khach) + transactions.dat (chi tiet don)
function loadCustomerData() {
  _custData = {};
  loadMenu();

  // 1. Load danh sach khach tu users.dat
  const users = loadUsersDat();
  users.forEach((u, idx) => {
    _custData[u.phone] = {
      userIdx: idx,
      name: u.name || '',
      desc: u.desc || '',
      count: u.totalOrders,
      totalSpent: 0,
      txns: []
    };
  });

  // 2. Load transactions binary va attach vao tung khach theo userIdx
  const allTxns = loadTransactionsDat();
  const phoneByIdx = users.map((u) => u.phone);
  allTxns.forEach((t) => {
    const phone = phoneByIdx[t.userIdx];
    if (!phone || !_custData[phone]) return;
    _custData[phone].totalSpent += t.total;
    _custData[phone].txns.push({
      ts: t.time,
      sess: t.session,
      items: t.items,
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

ipc.on('ready', (e) => {
  ready = true; port = e.port;
  activityLog.log(`{green-fg}${nowStr()} Server ready · ${e.menuItems} items · ${e.savedUsers} users saved{/}`);
  redraw();
});
ipc.on('session_opened', (e) => {
  sessionOpen = true; sessionCode = e.code; sessionStart = e.dateTime; codeInput = '';
  activityLog.log(`{green-fg}${nowStr()} Session OPENED code=${e.code}{/}`);
  redraw();
});
ipc.on('session_closed', (e) => {
  sessionOpen = false;
  activityLog.log(`{yellow-fg}${nowStr()} Session CLOSED · ${e.totalOrders} orders · revenue ${money(e.totalRevenue)}{/}`);
  redraw();
  setTimeout(() => { ipc.quit(); process.exit(0); }, 2500);
});
ipc.on('client_joined', (e) => {
  activityLog.log(`{cyan-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} CONNECTED{/}`);
  redraw();
});
ipc.on('client_left', (e) => {
  activityLog.log(`{gray-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} disconnected{/}`);
  redraw();
});
ipc.on('user_login', (e) => {
  const who = e.isNew ? '{green-fg}(NEW CUSTOMER){/}' : `{green-fg}(${e.orderCount} prior orders){/}`;
  activityLog.log(`{blue-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} login ${e.phone}{/} ${who}`);
  stats.totalSuggest++;
  redraw();
});
ipc.on('user_register', (e) => {
  activityLog.log(`{magenta-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} REGISTER ${e.phone} = "${e.name}"{/}`);
  redraw();
});
ipc.on('item_added', (e) => {
  activityLog.log(`{magenta-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} added {yellow-fg}${e.code}{/}{/}`);
  stats.totalSuggest++;
  redraw();
});
ipc.on('order_submitted', (e) => {
  const disc = e.discount > 0 ? ` {yellow-fg}(-${money(e.discount)} off){/}` : '';
  activityLog.log(`{green-fg}${nowStr()} Table ${String(e.slot + 1).padStart(2, '0')} SUBMIT #${e.orderId} · {white-fg}${money(e.total)}{/}${disc}{/}`);
  redraw();
});
ipc.on('stats', (e) => {
  stats.totalOrdersToday = e.totalOrdersToday;
  stats.revenueToday = e.revenueToday;
  stats.discountToday = e.discountToday;
  stats.discountedOrders = e.discountedOrders;
  stats.clientsConnected = e.clientsConnected;
  stats.usersKnown = e.usersKnown;
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
