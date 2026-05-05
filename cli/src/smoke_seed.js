// smoke_seed.js — Verify seed data: login voi SDT quen → suggest phai khac 0 va match preference.
// Khong confirm order (chi kiem tra suggest).

import { createIpc } from './ipc.js';

const TEST_PHONES = [
  { phone: '0901234567', expect: ['P01', 'D01'], name: 'Anh Nam  (Pho + Tra Da)' },
  { phone: '0923456789', expect: ['B01', 'T01'], name: 'Bac Hung (Bun + Che)   ' },
  { phone: '0956789012', expect: ['P02', 'T01'], name: 'Co Tu    (Pho Ga + Che)' },
  { phone: '0990123456', expect: ['P01'],        name: 'Anh Khoa (cold-start)  ' }
];

async function testOne({ phone, expect, name }) {
  return new Promise((resolve) => {
    const ipc = createIpc({ serverIp: '127.0.0.1', clientId: 1 });
    let suggested = null;
    let acked = null;

    ipc.on('session_start', () => ipc.send({ cmd: 'login', phone }));
    ipc.on('user_ack', (e) => { acked = e; });
    ipc.on('suggest', (e) => {
      if (suggested) return;
      suggested = e.items;
      const codes = suggested.map((x) => x.code);
      const match = expect.some((c) => codes.includes(c));
      const line = `${phone} ${name}: top3=${codes.join(',')}  scores=${suggested.map(x=>x.score.toFixed(2)).join(',')}  (expect includes ${expect.join('|')}) → ${match ? 'PASS' : 'FAIL'}`;
      console.log(line);
      console.log(`  user_ack: userId=${acked?.userId} isNew=${acked?.isNew} orderCount=${acked?.orderCount}`);
      ipc.quit();
    });

    ipc.onExit(() => resolve());
    setTimeout(() => { ipc.kill(); resolve(); }, 5000);
  });
}

(async () => {
  for (const t of TEST_PHONES) {
    await testOne(t);
    await new Promise((r) => setTimeout(r, 300));
  }
  process.exit(0);
})();
