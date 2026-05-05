// E2E: dang ky user moi + dat don -> verify users.dat & transactions.dat duoc ghi NGAY
// (khong cho den khi close session). Dashboard se doc duoc lich su ngay.
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const BASE = 'e:/code/project/DUT_PBL1/pbl1_recommendation_system_using_lantent_factor';

function log(c, m) {
  const col = { y: '\x1b[33m', g: '\x1b[32m', r: '\x1b[31m', b: '\x1b[36m' }[c] || '';
  process.stdout.write(`${col}${m}\x1b[0m\n`);
}

function readCString(buf, off, max) {
  const slice = buf.slice(off, off + max);
  const z = slice.indexOf(0);
  return slice.toString('utf8', 0, z === -1 ? max : z);
}

function spawnJson(bin, args, label, events) {
  const p = spawn(bin, args, { cwd: BASE });
  let buf = '';
  p.stdout.on('data', (d) => {
    buf += d.toString();
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line) continue;
      try { const e = JSON.parse(line); events.push(e); log('b', `[${label}] ${JSON.stringify(e).slice(0, 200)}`); }
      catch {}
    }
  });
  p.stderr.on('data', (d) => process.stderr.write(`${label}-ERR: ${d}`));
  return p;
}

const waitFor = async (events, ev, ms = 3000, pred = null) => {
  const s = Date.now();
  while (Date.now() - s < ms) {
    const e = events.find((x) => x.event === ev && (!pred || pred(x)));
    if (e) return e;
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error(`Timeout: ${ev}`);
};

function txnCount() {
  try {
    const buf = fs.readFileSync(path.join(BASE, 'data/transactions.dat'));
    return buf.readInt32LE(0);
  } catch { return -1; }
}

function userTotalOrders(phone) {
  try {
    const buf = fs.readFileSync(path.join(BASE, 'data/users.dat'));
    const n = buf.readInt32LE(0);
    let off = 4;
    const phones = [];
    for (let i = 0; i < n; i++) phones.push(readCString(buf, off + i*11, 10));
    off += n * 11 + n * 40 + n * 80;  // skip name + desc
    const idx = phones.indexOf(phone);
    if (idx < 0) return -1;
    return buf.readInt32LE(off + idx * 4);
  } catch (e) { return -1; }
}

async function main() {
  // Reseed
  log('y', '[Seed]');
  for (const f of ['users.dat', 'transactions.dat', 'lfm_P.dat', 'lfm_Q.dat', 'transactions.log']) {
    try { fs.unlinkSync(path.join(BASE, 'data', f)); } catch {}
  }
  await new Promise((res, rej) => {
    const p = spawn(path.join(BASE, 'build/seed_data.exe'), [], { cwd: BASE });
    p.on('exit', (c) => c === 0 ? res() : rej(new Error('seed fail')));
  });
  const txnBefore = txnCount();
  log('g', `  txn count after seed: ${txnBefore}`);

  // Server + client
  log('y', '[Start server + client]');
  const srvEvents = []; const cliEvents = [];
  const srv = spawnJson(path.join(BASE, 'build/server.exe'), ['--server', '--json'], 'srv', srvEvents);
  await waitFor(srvEvents, 'ready');
  srv.stdin.write(JSON.stringify({ cmd: 'open_session', code: '9999' }) + '\n');
  await waitFor(srvEvents, 'session_opened');

  const cli = spawnJson(path.join(BASE, 'build/client.exe'), ['--client', '127.0.0.1', '7', '--json'], 'cli', cliEvents);
  await waitFor(cliEvents, 'connected');
  await waitFor(cliEvents, 'session_start');

  // Register new user
  const PHONE = '0911222333';
  log('y', `[Register new user ${PHONE}]`);
  cli.stdin.write(JSON.stringify({ cmd: 'login', phone: PHONE }) + '\n');
  await waitFor(cliEvents, 'user_ack', 3000, (e) => e.isNew === true);
  cli.stdin.write(JSON.stringify({ cmd: 'register', phone: PHONE, name: 'Khach Moi', desc: 'Demo test' }) + '\n');
  await waitFor(cliEvents, 'user_ack', 3000, (e) => e.isNew === false && e.name === 'Khach Moi');
  await waitFor(cliEvents, 'suggest');
  log('g', `  user_total_orders(${PHONE}) immediately after register = ${userTotalOrders(PHONE)}`);

  // Place order
  log('y', '[Place order: P01 x2, D01 x1]');
  cliEvents.length = 0;
  cli.stdin.write(JSON.stringify({ cmd: 'add_item', code: 'P01', qty: 2 }) + '\n');
  await new Promise(r => setTimeout(r, 200));
  cli.stdin.write(JSON.stringify({ cmd: 'add_item', code: 'D01', qty: 1 }) + '\n');
  await new Promise(r => setTimeout(r, 200));
  cli.stdin.write(JSON.stringify({ cmd: 'finish' }) + '\n');
  await waitFor(cliEvents, 'invoice_ready');
  cli.stdin.write(JSON.stringify({ cmd: 'confirm' }) + '\n');
  await waitFor(cliEvents, 'order_ack');

  // Verify IMMEDIATELY (no session close)
  await new Promise(r => setTimeout(r, 300));   // small settle
  const txnAfter = txnCount();
  const totalOrdersAfter = userTotalOrders(PHONE);
  log('g', `  After order submit (no close session):`);
  log('g', `    transactions.dat count: ${txnBefore} -> ${txnAfter} (expected +1)`);
  log('g', `    users.dat[${PHONE}].totalOrders = ${totalOrdersAfter} (expected 1)`);

  if (txnAfter !== txnBefore + 1) {
    throw new Error(`FAIL: txnCount should have increased by 1, got ${txnBefore} -> ${txnAfter}`);
  }
  if (totalOrdersAfter !== 1) {
    throw new Error(`FAIL: userTotalOrders should be 1, got ${totalOrdersAfter}`);
  }
  log('g', '  PASS: data persisted to disk immediately after order submit');

  cli.kill();
  srv.stdin.write(JSON.stringify({ cmd: 'close_session', code: '9999' }) + '\n');
  await waitFor(srvEvents, 'session_closed');
  srv.kill();
  log('g', '\n=== TEST PASS ===');
  process.exit(0);
}

main().catch((e) => { log('r', 'FAIL: ' + e.message); console.error(e.stack); process.exit(1); });
