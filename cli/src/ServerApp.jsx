// ServerApp.jsx — React Ink dashboard cho thu ngan
// Chay: cd cli && npm run server

import React, { useEffect, useMemo, useState } from 'react';
import { render, Box, Text, useApp, useInput } from 'ink';
import BigText from 'ink-big-text';
import Gradient from 'ink-gradient';
import Spinner from 'ink-spinner';
import { createServerIpc } from './server_ipc.js';

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}
function nowStr() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

const MAX_CLIENTS = 20;
const MAX_LOG = 12;

function Header({ port, ready }) {
  return (
    <Box flexDirection="column" alignItems="flex-start">
      <Gradient name="rainbow">
        <BigText text="Viet Phong" font="tiny" />
      </Gradient>
      <Box>
        <Text color="gray">Restaurant Dashboard (Cashier)  · </Text>
        {ready
          ? <Text color="green">● Server READY</Text>
          : <Text color="yellow"><Spinner type="dots" /> starting...</Text>}
        {port ? <Text color="gray">  · port {port}</Text> : null}
      </Box>
    </Box>
  );
}

function SessionPanel({ sessionOpen, sessionCode, sessionStart, codeInput }) {
  return (
    <Box flexDirection="column" borderStyle="round" borderColor={sessionOpen ? 'green' : 'yellow'} paddingX={1} width={46}>
      <Text bold color={sessionOpen ? 'green' : 'yellow'}>
        {sessionOpen ? '● SESSION OPEN' : '○ SESSION CLOSED'}
      </Text>
      {sessionOpen ? (
        <>
          <Text>Code   : <Text color="cyan">{sessionCode}</Text></Text>
          <Text>Started: <Text color="cyan">{sessionStart}</Text></Text>
          <Box marginTop={1}>
            <Text dimColor>Re-enter code + Enter to CLOSE:</Text>
          </Box>
          <Box>
            <Text>&gt; <Text color="cyan" bold>{codeInput}</Text><Text color="gray">_</Text></Text>
          </Box>
        </>
      ) : (
        <>
          <Text dimColor>Enter CODE (1-9 digits) + Enter to OPEN:</Text>
          <Box marginTop={1}>
            <Text>&gt; <Text color="cyan" bold>{codeInput}</Text><Text color="gray">_</Text></Text>
          </Box>
        </>
      )}
    </Box>
  );
}

function StatsPanel({ stats }) {
  const accept = stats.totalSuggest > 0
    ? Math.round(100 * stats.suggestAccepted / stats.totalSuggest) : 0;

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="blue" paddingX={1} width={46}>
      <Text bold color="blue">TODAY'S STATS</Text>
      <Text>Orders: <Text bold>{stats.totalOrdersToday}</Text>  |  Revenue: <Text color="green" bold>{money(stats.revenueToday)}</Text></Text>
      <Text>Discount: <Text color="yellow">{money(stats.discountToday)}</Text>  |  Discounted: <Text color="yellow">{stats.discountedOrders}</Text></Text>
      <Text>Tables: <Text color="cyan">{stats.clientsConnected}</Text>/{MAX_CLIENTS}  ·  Users: <Text color="cyan">{stats.usersKnown}</Text></Text>
      <Text>LFM suggest: <Text color="magenta">{stats.suggestAccepted}</Text>/{stats.totalSuggest}  ({accept}%)</Text>
    </Box>
  );
}

function ClientsPanel({ slots }) {
  const active = slots.filter((s) => s.active);
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1} width={52}>
      <Text bold color="cyan">TABLES ({active.length} connected)</Text>
      {active.length === 0
        ? <Text dimColor>No clients connected yet.</Text>
        : active.map((s) => (
            <Text key={s.slot}>
              <Text color="yellow">Table {String(s.slot + 1).padStart(2, '0')}</Text>
              {'  '}
              {s.phone
                ? <><Text color="green">{s.phone}</Text> <Text dimColor>({s.orderCount} orders)</Text></>
                : <Text dimColor>(not logged in)</Text>}
              {'  '}
              <Text color={s.state === 'ORDERING' ? 'magenta' : (s.state === 'CHECKOUT' ? 'yellow' : 'gray')}>
                [{s.state}]
              </Text>
            </Text>
          ))
      }
    </Box>
  );
}

function ActivityLog({ entries }) {
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} flexGrow={1}>
      <Text bold>ACTIVITY LOG</Text>
      {entries.length === 0 ? <Text dimColor>(no activity yet)</Text> : null}
      {entries.slice(0, MAX_LOG).map((e, i) => (
        <Text key={i}>
          <Text dimColor>{e.t}</Text>{' '}
          <Text color={e.color}>{e.line}</Text>
        </Text>
      ))}
    </Box>
  );
}

function App() {
  const { exit } = useApp();
  const ipc = useMemo(() => createServerIpc(), []);

  const [ready, setReady] = useState(false);
  const [port, setPort] = useState(8888);
  const [sessionOpen, setSessionOpen] = useState(false);
  const [sessionCode, setSessionCode] = useState('');
  const [sessionStart, setSessionStart] = useState('');
  const [codeInput, setCodeInput] = useState('');
  const [slots, setSlots] = useState(
    Array.from({ length: MAX_CLIENTS }, (_, i) => ({ slot: i, active: false, phone: null, orderCount: 0, state: 'IDLE' }))
  );
  const [stats, setStats] = useState({
    totalOrdersToday: 0, revenueToday: 0, discountToday: 0, discountedOrders: 0,
    clientsConnected: 0, usersKnown: 0,
    totalSuggest: 0, suggestAccepted: 0
  });
  const [log, setLog] = useState([]);
  const [error, setError] = useState('');

  const push = (line, color = 'white') =>
    setLog((L) => [{ t: nowStr(), line, color }, ...L].slice(0, MAX_LOG));

  useEffect(() => {
    const unsubs = [];
    unsubs.push(ipc.on('ready', (e) => {
      setReady(true);
      setPort(e.port);
      push(`Server ready · ${e.menuItems} items · ${e.savedUsers} users saved`, 'green');
    }));
    unsubs.push(ipc.on('session_opened', (e) => {
      setSessionOpen(true);
      setSessionCode(e.code);
      setSessionStart(e.dateTime);
      setCodeInput('');
      push(`Session OPENED code=${e.code}`, 'green');
    }));
    unsubs.push(ipc.on('session_closed', (e) => {
      setSessionOpen(false);
      push(`Session CLOSED · ${e.totalOrders} orders · revenue ${money(e.totalRevenue)}`, 'yellow');
      setTimeout(() => { ipc.quit(); exit(); }, 2000);
    }));
    unsubs.push(ipc.on('client_joined', (e) => {
      setSlots((S) => S.map((x) => x.slot === e.slot ? { ...x, active: true, state: 'IDLE' } : x));
      push(`Table ${String(e.slot + 1).padStart(2, '0')} CONNECTED`, 'cyan');
    }));
    unsubs.push(ipc.on('client_left', (e) => {
      setSlots((S) => S.map((x) => x.slot === e.slot ? { slot: e.slot, active: false, phone: null, orderCount: 0, state: 'IDLE' } : x));
      push(`Table ${String(e.slot + 1).padStart(2, '0')} disconnected`, 'gray');
    }));
    unsubs.push(ipc.on('user_login', (e) => {
      setSlots((S) => S.map((x) => x.slot === e.slot ? { ...x, phone: e.phone, orderCount: e.orderCount, state: 'ORDERING' } : x));
      push(`Table ${String(e.slot + 1).padStart(2, '0')} login ${e.phone} ${e.isNew ? '(new customer)' : `(${e.orderCount} prior orders)`}`, 'blue');
      setStats((st) => ({ ...st, totalSuggest: st.totalSuggest + 1 }));
    }));
    unsubs.push(ipc.on('item_added', (e) => {
      push(`Table ${String(e.slot + 1).padStart(2, '0')} added ${e.code}`, 'magenta');
      setStats((st) => ({ ...st, totalSuggest: st.totalSuggest + 1 }));
    }));
    unsubs.push(ipc.on('order_submitted', (e) => {
      setSlots((S) => S.map((x) => x.slot === e.slot ? { ...x, state: 'IDLE', phone: null, orderCount: 0 } : x));
      const disc = e.discount > 0 ? ` (-${money(e.discount)} off)` : '';
      push(`Table ${String(e.slot + 1).padStart(2, '0')} SUBMIT #${e.orderId} · ${money(e.total)}${disc}`, 'green');
    }));
    unsubs.push(ipc.on('stats', (e) => {
      setStats((st) => ({ ...st,
        totalOrdersToday: e.totalOrdersToday,
        revenueToday: e.revenueToday,
        discountToday: e.discountToday,
        discountedOrders: e.discountedOrders,
        clientsConnected: e.clientsConnected,
        usersKnown: e.usersKnown
      }));
    }));
    unsubs.push(ipc.on('error', (e) => setError(e.message || 'unknown error')));
    unsubs.push(ipc.onExit(() => push('server.exe exited', 'red')));
    return () => unsubs.forEach((u) => u && u());
  }, [ipc, exit]);

  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) { ipc.quit(); exit(); return; }

    if (key.return) {
      if (codeInput.length === 0) return;
      if (!sessionOpen) ipc.openSession(codeInput);
      else ipc.closeSession(codeInput);
      return;
    }
    if (key.backspace || key.delete) { setCodeInput((s) => s.slice(0, -1)); setError(''); return; }
    if (input && /^\d$/.test(input) && codeInput.length < 9) {
      setCodeInput((s) => s + input);
      setError('');
    }
  });

  return (
    <Box flexDirection="column" paddingX={1}>
      <Header port={port} ready={ready} />

      <Box flexDirection="row" marginTop={1}>
        <SessionPanel
          sessionOpen={sessionOpen}
          sessionCode={sessionCode}
          sessionStart={sessionStart}
          codeInput={codeInput}
        />
        <Box marginLeft={1}>
          <StatsPanel stats={stats} />
        </Box>
      </Box>

      <Box flexDirection="row" marginTop={1}>
        <ClientsPanel slots={slots} />
        <Box marginLeft={1} flexGrow={1}>
          <ActivityLog entries={log} />
        </Box>
      </Box>

      {error ? (
        <Box marginTop={1}><Text color="red" bold>Error: {error}</Text></Box>
      ) : null}

      <Box marginTop={1}>
        <Text dimColor>
          {sessionOpen
            ? 'Re-enter code + Enter to CLOSE  ·  Esc/Ctrl-C to quit'
            : 'Enter CODE + Enter to OPEN  ·  Esc/Ctrl-C to quit'}
        </Text>
      </Box>
    </Box>
  );
}

render(<App />);
