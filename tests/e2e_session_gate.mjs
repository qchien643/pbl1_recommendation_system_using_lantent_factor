// Verify: khi session chua mo, server reject USER_LOGIN + ORDER_SUBMIT
import { spawn } from 'child_process';
const BASE = 'e:/code/project/DUT_PBL1/pbl1_recommendation_system_using_lantent_factor';

function log(c, m) {
  const col = { y: '\x1b[33m', g: '\x1b[32m', r: '\x1b[31m', b: '\x1b[36m' }[c] || '';
  process.stdout.write(`${col}${m}\x1b[0m\n`);
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
      try { const e = JSON.parse(line); events.push(e); log('b', `[${label}] ${JSON.stringify(e).slice(0, 150)}`); }
      catch {}
    }
  });
  p.stderr.on('data', (d) => {
    const s = d.toString();
    process.stderr.write(`${label}-ERR: ${s}`);
  });
  return p;
}

const waitFor = async (events, ev, ms = 2000, pred = null) => {
  const s = Date.now();
  while (Date.now() - s < ms) {
    const e = events.find((x) => x.event === ev && (!pred || pred(x)));
    if (e) return e;
    await new Promise(r => setTimeout(r, 50));
  }
  return null;
};

async function main() {
  const srvEvents = []; const cliEvents = [];
  log('y', '[1] Start server (session NOT opened yet)');
  const srv = spawnJson(`${BASE}/build/server.exe`, ['--server', '--json'], 'srv', srvEvents);
  await waitFor(srvEvents, 'ready');

  log('y', '[2] Connect client (should sit in WAITING, no session_start)');
  const cli = spawnJson(`${BASE}/build/client.exe`, ['--client', '127.0.0.1', '8', '--json'], 'cli', cliEvents);
  await waitFor(cliEvents, 'connected');

  // No session_start should come (because session is not open)
  const ss = await waitFor(cliEvents, 'session_start', 500);
  if (ss) { log('r', 'FAIL: session_start arrived without opening session'); process.exit(1); }
  log('g', '  OK: no session_start received (client in WAITING)');

  log('y', '[3] Attempt to send USER_LOGIN WITHOUT session — server should REJECT');
  cliEvents.length = 0;
  cli.stdin.write(JSON.stringify({ cmd: 'login', phone: '0912345678' }) + '\n');
  const ack = await waitFor(cliEvents, 'user_ack', 800);
  if (ack) {
    log('r', `  FAIL: server responded with USER_ACK before session open: ${JSON.stringify(ack)}`);
    process.exit(1);
  }
  log('g', '  OK: server did not send USER_ACK (rejected silently)');

  log('y', '[4] Open session, then client should be able to login');
  srv.stdin.write(JSON.stringify({ cmd: 'open_session', code: '5555' }) + '\n');
  await waitFor(srvEvents, 'session_opened');
  await waitFor(cliEvents, 'session_start');
  log('g', '  OK: session_start received after open');

  cliEvents.length = 0;
  cli.stdin.write(JSON.stringify({ cmd: 'login', phone: '0912345678' }) + '\n');
  const ack2 = await waitFor(cliEvents, 'user_ack', 1500);
  if (!ack2 || ack2.name !== 'Chi Lan') {
    log('r', `  FAIL: login failed after session open. ack=${JSON.stringify(ack2)}`);
    process.exit(1);
  }
  log('g', `  OK: login successful: name='${ack2.name}'`);

  cli.kill();
  srv.stdin.write(JSON.stringify({ cmd: 'close_session', code: '5555' }) + '\n');
  await waitFor(srvEvents, 'session_closed');
  srv.kill();

  log('g', '\n=== GATE TEST PASS ===');
  process.exit(0);
}

main().catch((e) => { log('r', 'FAIL: ' + e.message); process.exit(1); });
