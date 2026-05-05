// ipc.js — Spawn C++ client (--json mode) va wrap stdio IPC
// Events flow: C++ stdout -> parse JSON line -> callback
// Commands  : cmd object -> JSON.stringify + '\n' -> C++ stdin

import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createIpc({ serverIp = '127.0.0.1', clientId = 1, binary = null } = {}) {
  // Resolve binary: default to ../../build/client.exe relative to this file
  const exe = binary || path.resolve(__dirname, '../../build/client.exe');

  const child = spawn(exe, ['--client', serverIp, String(clientId), '--json'], {
    stdio: ['pipe', 'pipe', 'pipe']
  });

  const listeners = new Map();   // event name -> Set<callback>
  const anyListeners = new Set(); // callbacks receiving ALL events

  let stdoutBuf = '';
  child.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString('utf8');
    let nl;
    while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
      const line = stdoutBuf.slice(0, nl).trim();
      stdoutBuf = stdoutBuf.slice(nl + 1);
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); }
      catch { continue; }
      if (!ev || !ev.event) continue;
      const cbs = listeners.get(ev.event);
      if (cbs) for (const cb of cbs) cb(ev);
      for (const cb of anyListeners) cb(ev);
    }
  });

  child.stderr.on('data', (chunk) => {
    // Forward to our own stderr for debugging (not UI)
    process.stderr.write('[client.exe] ' + chunk.toString('utf8'));
  });

  const exitListeners = new Set();
  child.on('exit', (code) => {
    for (const cb of exitListeners) cb(code);
  });

  return {
    on(eventName, cb) {
      if (!listeners.has(eventName)) listeners.set(eventName, new Set());
      listeners.get(eventName).add(cb);
      return () => listeners.get(eventName)?.delete(cb);
    },
    onAny(cb) {
      anyListeners.add(cb);
      return () => anyListeners.delete(cb);
    },
    onExit(cb) { exitListeners.add(cb); return () => exitListeners.delete(cb); },
    send(cmdObj) {
      if (!child.stdin.writable) return false;
      child.stdin.write(JSON.stringify(cmdObj) + '\n');
      return true;
    },
    quit() {
      try { child.stdin.write(JSON.stringify({ cmd: 'quit' }) + '\n'); } catch {}
      try { child.stdin.end(); } catch {}
    },
    kill() { child.kill(); }
  };
}
