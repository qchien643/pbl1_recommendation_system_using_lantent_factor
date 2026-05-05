// pretty_event.js — Doc JSON event lines tren stdin, in ra format de doc cho demo.
// Usage: ... --json | node tools/pretty_event.js "KHACH 1"

import readline from 'node:readline';

const label = process.argv[2] || 'CLIENT';
const rl = readline.createInterface({ input: process.stdin });

function money(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd'; }
function bar(score, max = 10) {
  if (!Number.isFinite(score) || max <= 0) return '.'.repeat(10);
  const n = Math.max(0, Math.min(10, Math.round((score / max) * 10)));
  return '#'.repeat(n) + '.'.repeat(10 - n);
}

let maxScore = 0;

rl.on('line', (line) => {
  line = line.trim();
  if (!line) return;
  let ev;
  try { ev = JSON.parse(line); } catch { return; }
  if (!ev.event) return;

  switch (ev.event) {
    case 'connected':
      console.log(`  [${label}] >> Connected to server`);
      break;
    case 'session_start':
      console.log(`  [${label}] >> Session START: code=${ev.code}`);
      break;
    case 'menu':
      console.log(`  [${label}] >> Menu loaded (${ev.items.length} mon)`);
      break;
    case 'user_ack':
      if (ev.isNew) {
        console.log(`  [${label}] >> KHACH MOI — userId=${ev.userId}`);
      } else {
        console.log(`  [${label}] >> Chao mung tro lai — userId=${ev.userId}, da dat ${ev.orderCount} don truoc.`);
      }
      break;
    case 'suggest':
      maxScore = Math.max(...ev.items.map(x => x.score || 0), 0.001);
      console.log(`  [${label}] >> GOI Y LFM:`);
      ev.items.forEach((it, i) => {
        console.log(`      ${i+1}. ${it.code.padEnd(4)} ${(it.name||'').padEnd(24)} ${bar(it.score, maxScore)}  ${it.score.toFixed(3)}`);
      });
      break;
    case 'invoice_ready':
      console.log(`  [${label}] >> HOA DON:`);
      ev.items.forEach((it) => {
        console.log(`      ${it.code} ${(it.name||'').padEnd(22)} x${String(it.qty).padStart(2)}  ${money(it.price).padStart(10)}  =  ${money(it.subtotal).padStart(12)}`);
      });
      console.log(`      Tam tinh: ${money(ev.subtotal)}`);
      if (ev.discount > 0) {
        console.log(`      Giam gia (25%, don >= 2tr): -${money(ev.discount)}`);
      }
      console.log(`      TONG CONG: ${money(ev.total)}`);
      break;
    case 'order_ack':
      if (ev.status === 'OK') {
        console.log(`  [${label}] >> ORDER_ACK: OK — orderId=${ev.orderId}`);
      } else {
        console.log(`  [${label}] >> ORDER_ACK: FAIL`);
      }
      break;
    case 'session_stop':
      console.log(`  [${label}] >> Session STOP`);
      break;
    case 'disconnected':
      console.log(`  [${label}] >> Disconnected`);
      break;
    default:
      console.log(`  [${label}] >> ${ev.event}`);
  }
});
