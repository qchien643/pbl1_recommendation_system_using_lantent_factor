// E2E test: server + client JSON mode, test login existing + register new user + persistence
// Run: node tests/e2e_register.mjs
import { spawn } from 'child_process';

const BASE = 'e:/code/project/DUT_PBL1/pbl1_recommendation_system_using_lantent_factor';

function log(color, msg) {
  const col = { y: '\x1b[33m', g: '\x1b[32m', c: '\x1b[36m', r: '\x1b[31m' }[color] || '';
  process.stdout.write(`${col}${msg}\x1b[0m\n`);
}

function spawnJsonProc(bin, args, label, events) {
  const proc = spawn(bin, args, { cwd: BASE });
  let buf = '';
  proc.stdout.on('data', (d) => {
    buf += d.toString();
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        const e = JSON.parse(line);
        events.push(e);
        log('c', `[${label}] ${JSON.stringify(e)}`);
      } catch {}
    }
  });
  proc.stderr.on('data', (d) => process.stderr.write(`${label}-ERR: ${d}`));
  return proc;
}

const waitFor = async (events, evName, timeoutMs = 3000, predicate = null) => {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const e = events.find((x) => x.event === evName && (!predicate || predicate(x)));
    if (e) return e;
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error(`Timeout waiting for ${evName}`);
};

async function main() {
  // ---- Phase 1: seed existing + new user register ----
  log('y', '[Phase 1] Start server');
  const srvEvents = [];
  const srv = spawnJsonProc(`${BASE}/build/server.exe`, ['--server', '--json'], 'srv', srvEvents);
  await waitFor(srvEvents, 'ready');

  log('y', '[Phase 1] Open session 7777');
  srv.stdin.write(JSON.stringify({ cmd: 'open_session', code: '7777' }) + '\n');
  await waitFor(srvEvents, 'session_opened');

  log('y', '[Phase 1] Start client 5');
  const cliEvents = [];
  const cli = spawnJsonProc(`${BASE}/build/client.exe`, ['--client', '127.0.0.1', '5', '--json'], 'cli', cliEvents);
  await waitFor(cliEvents, 'connected');
  await waitFor(cliEvents, 'session_start');

  // Test 1: returning user "Anh Nam"
  log('y', '[T1] Returning user 0901234567');
  cli.stdin.write(JSON.stringify({ cmd: 'login', phone: '0901234567' }) + '\n');
  const ack1 = await waitFor(cliEvents, 'user_ack');
  if (ack1.isNew !== false) throw new Error(`T1: expected isNew=false, got ${ack1.isNew}`);
  if (ack1.name !== 'Anh Nam') throw new Error(`T1: expected name='Anh Nam', got '${ack1.name}'`);
  log('g', `  T1 PASS: name='${ack1.name}', isNew=${ack1.isNew}`);

  cliEvents.length = 0;

  // Test 2: new user 0900000099
  log('y', '[T2] New user 0900000099 login');
  cli.stdin.write(JSON.stringify({ cmd: 'login', phone: '0900000099' }) + '\n');
  const ack2 = await waitFor(cliEvents, 'user_ack');
  if (ack2.isNew !== true) throw new Error(`T2: expected isNew=true, got ${ack2.isNew}`);
  if (ack2.name !== '') throw new Error(`T2: expected empty name, got '${ack2.name}'`);
  log('g', `  T2 PASS: isNew=true, name=''`);

  log('y', '[T2] Send register: name="Test User"');
  cli.stdin.write(JSON.stringify({ cmd: 'register', phone: '0900000099', name: 'Test User', desc: 'Khach test' }) + '\n');
  const ack3 = await waitFor(cliEvents, 'user_ack', 3000, (e) => e.isNew === false && e.name === 'Test User');
  log('g', `  T2 PASS post-register: name='${ack3.name}' isNew=${ack3.isNew}`);
  await waitFor(cliEvents, 'suggest');
  log('g', '  T2 PASS: got suggest after register');

  log('y', '[Phase 1] Close session');
  srv.stdin.write(JSON.stringify({ cmd: 'close_session', code: '7777' }) + '\n');
  await waitFor(srvEvents, 'session_closed');
  cli.kill();
  srv.kill();
  await new Promise(r => setTimeout(r, 500));

  // ---- Phase 2: verify persistence ----
  log('y', '[Phase 2] Restart server');
  const srv2Events = [];
  const srv2 = spawnJsonProc(`${BASE}/build/server.exe`, ['--server', '--json'], 'srv2', srv2Events);
  const ready2 = await waitFor(srv2Events, 'ready');
  if (ready2.savedUsers !== 11) {
    throw new Error(`T3: expected 11 users (10 seed + 1 new), got ${ready2.savedUsers}`);
  }
  log('g', `  T3 PASS: savedUsers=${ready2.savedUsers}`);

  srv2.stdin.write(JSON.stringify({ cmd: 'open_session', code: '8888' }) + '\n');
  await waitFor(srv2Events, 'session_opened');

  const cli2Events = [];
  const cli2 = spawnJsonProc(`${BASE}/build/client.exe`, ['--client', '127.0.0.1', '6', '--json'], 'cli2', cli2Events);
  await waitFor(cli2Events, 'connected');
  await waitFor(cli2Events, 'session_start');

  log('y', '[T4] Login persisted user 0900000099');
  cli2.stdin.write(JSON.stringify({ cmd: 'login', phone: '0900000099' }) + '\n');
  const ack4 = await waitFor(cli2Events, 'user_ack');
  if (ack4.isNew !== false || ack4.name !== 'Test User') {
    throw new Error(`T4: expected persisted name='Test User' isNew=false, got name='${ack4.name}' isNew=${ack4.isNew}`);
  }
  log('g', `  T4 PASS: persistence OK — name='${ack4.name}' isNew=${ack4.isNew}`);

  cli2.kill();
  srv2.stdin.write(JSON.stringify({ cmd: 'close_session', code: '8888' }) + '\n');
  await waitFor(srv2Events, 'session_closed');
  srv2.kill();

  log('g', '\n=== ALL TESTS PASS ===');
  process.exit(0);
}

main().catch((e) => {
  log('r', 'FAIL: ' + e.message);
  console.error(e.stack);
  process.exit(1);
});
