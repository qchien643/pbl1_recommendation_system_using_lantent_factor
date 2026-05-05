// smoke_test.js — Test IPC layer (Node <-> C++ client --json) end-to-end.
// Ham trong: spawn C++ --json, gui JSON cmds, log JSON events nhan ve.
// Khong can TTY — dung de CI smoke test.
//
// Usage:
//   node cli/src/smoke_test.js [serverIp] [clientId]

import { createIpc } from './ipc.js';

const args = process.argv.slice(2);
const serverIp = args[0] || '127.0.0.1';
const clientId = parseInt(args[1] || '1', 10);

const ipc = createIpc({ serverIp, clientId });

ipc.onAny((ev) => {
  console.log('EVENT<', JSON.stringify(ev));
});

function sendAfter(ms, cmd) {
  setTimeout(() => {
    console.log('CMD  >', JSON.stringify(cmd));
    ipc.send(cmd);
  }, ms);
}

// Scripted flow
sendAfter(800,  { cmd: 'login',    phone: '0901234567' });
sendAfter(1600, { cmd: 'add_item', code: 'P01', qty: 2 });
sendAfter(2200, { cmd: 'add_item', code: 'D01', qty: 1 });
sendAfter(2800, { cmd: 'finish' });
sendAfter(3400, { cmd: 'confirm' });
sendAfter(4500, { cmd: 'quit' });

ipc.onExit((code) => {
  console.log(`Child exited: ${code}`);
  process.exit(0);
});

setTimeout(() => { console.log('TIMEOUT after 10s'); ipc.kill(); process.exit(1); }, 10000);
