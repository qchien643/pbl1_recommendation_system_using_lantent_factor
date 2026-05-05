// server_ipc.js — Spawn build/server.exe --server --json + parse events
// Tuong tu ipc.js (client) nhung binary va commands khac.

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createServerIpc({ binary = null } = {}) {
  const exe = binary || path.resolve(__dirname, '../../build/server.exe');
  const child = spawn(exe, ['--server', '--json'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: path.resolve(__dirname, '../..')
  });

  const listeners = new Map();
  const anyListeners = new Set();
  let stdoutBuf = '';

  child.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString('utf8');
    let nl;
    while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
      const line = stdoutBuf.slice(0, nl).trim();
      stdoutBuf = stdoutBuf.slice(nl + 1);
      if (!line) continue;
      let ev; try { ev = JSON.parse(line); } catch { continue; }
      if (!ev?.event) continue;
      listeners.get(ev.event)?.forEach((cb) => cb(ev));
      anyListeners.forEach((cb) => cb(ev));
    }
  });

  child.stderr.on('data', (chunk) => {
    process.stderr.write('[server.exe] ' + chunk.toString('utf8'));
  });

  const exitListeners = new Set();
  child.on('exit', (code) => exitListeners.forEach((cb) => cb(code)));

  return {
    on(ev, cb) {
      if (!listeners.has(ev)) listeners.set(ev, new Set());
      listeners.get(ev).add(cb);
      return () => listeners.get(ev)?.delete(cb);
    },
    onAny(cb) { anyListeners.add(cb); return () => anyListeners.delete(cb); },
    onExit(cb) { exitListeners.add(cb); return () => exitListeners.delete(cb); },
    send(cmdObj) {
      if (!child.stdin.writable) return false;
      child.stdin.write(JSON.stringify(cmdObj) + '\n');
      return true;
    },
    openSession(code) { return this.send({ cmd: 'open_session', code }); },
    closeSession(code) { return this.send({ cmd: 'close_session', code }); },
    getStats() { return this.send({ cmd: 'get_stats' }); },
    quit() {
      try { this.send({ cmd: 'quit' }); child.stdin.end(); } catch {}
    },
    kill() { child.kill(); }
  };
}
