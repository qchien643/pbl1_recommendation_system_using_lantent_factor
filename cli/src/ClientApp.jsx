// ClientApp.jsx — React Ink UI cho may ban khach
// Chay: cd cli && npm run client -- [serverIp] [clientId]

import React, { useEffect, useState, useMemo } from 'react';
import { render, Box, Text, useApp, useInput } from 'ink';
import BigText from 'ink-big-text';
import Gradient from 'ink-gradient';
import Spinner from 'ink-spinner';
import { createIpc } from './ipc.js';

import WaitingScreen  from './components/WaitingScreen.jsx';
import PhoneInput     from './components/PhoneInput.jsx';
import NameInput      from './components/NameInput.jsx';
import MenuDisplay    from './components/MenuDisplay.jsx';
import SuggestPanel   from './components/SuggestPanel.jsx';
import OrderSummary   from './components/OrderSummary.jsx';
import Invoice        from './components/Invoice.jsx';
import DailySummary   from './components/DailySummary.jsx';

const S = {
  CONNECTING:  'connecting',
  WAITING:     'waiting',
  PHONE:       'phone',
  LOADING:     'loading',
  NAME_INPUT:  'name_input',     // khach moi nhap ten + desc
  REGISTERING: 'registering',    // dang cho server ack sau USER_REGISTER
  ORDERING:    'ordering',
  CHECKOUT:    'checkout',
  SUBMITTING:  'submitting',
  THANKS:      'thanks',
  CLOSED:      'closed',
  ERROR:       'error'
};

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}

function TitleBar({ clientId, sessionCode, phone, state }) {
  const statusLabel = {
    connecting:  'KẾT NỐI',
    waiting:     'CHỜ CA MỞ',
    phone:       'NHẬP SĐT',
    loading:     'ĐANG TẢI',
    name_input:  'ĐĂNG KÝ',
    registering: 'LƯU TÊN',
    ordering:    'ĐẶT MÓN',
    checkout:    'XÁC NHẬN',
    submitting:  'GỬI ĐƠN',
    thanks:      'HOÀN TẤT',
    closed:      'CA ĐÓNG',
    error:       'LỖI'
  }[state] || '';

  const statusColor = {
    connecting: 'yellow', waiting: 'yellow', phone: 'blue', loading: 'yellow',
    name_input: 'magenta', registering: 'yellow',
    ordering: 'magenta', checkout: 'cyan', submitting: 'yellow',
    thanks: 'green', closed: 'gray', error: 'red'
  }[state] || 'white';

  return (
    <Box flexDirection="column">
      <Gradient name="cristal">
        <BigText text={`Ban ${String(clientId).padStart(2, '0')}`} font="tiny" />
      </Gradient>
      <Box flexDirection="row">
        <Text color="yellow">◆ </Text>
        <Text color="gray">Nhà hàng Việt Phong  ·  </Text>
        {sessionCode ? <Text><Text color="cyan">● </Text>Mã GD: <Text color="cyan">{sessionCode}</Text>  ·  </Text> : null}
        {phone ? <Text><Text color="green">★ </Text>SĐT: <Text color="green">{phone}</Text>  ·  </Text> : null}
        <Text color={statusColor}><Spinner type="dots" /></Text>
        <Text color={statusColor} bold> [{statusLabel}]</Text>
      </Box>
    </Box>
  );
}

function OrderInput({ onAdd, onFinish, disabled }) {
  const [value, setValue] = useState('');
  const [flash, setFlash] = useState('');

  useInput((input, key) => {
    if (disabled) return;
    if (key.return) {
      const v = value.trim();
      setValue('');
      if (v === '' || v === '00') { onFinish(); return; }
      const m = v.match(/^([A-Za-z][0-9][0-9])\s*(\d+)?$/);
      if (!m) {
        setFlash('Nhập mã hợp lệ (VD: P01 hoặc P01 2)');
        setTimeout(() => setFlash(''), 1500);
        return;
      }
      const code = m[1].toUpperCase();
      const qty = m[2] ? parseInt(m[2], 10) : 1;
      if (qty <= 0) { setFlash('Số lượng phải > 0'); setTimeout(() => setFlash(''), 1500); return; }
      onAdd(code, qty);
      setFlash('');
      return;
    }
    if (key.backspace || key.delete) { setValue((v) => v.slice(0, -1)); return; }
    if (input && !key.ctrl && !key.meta) {
      if (value.length < 12) setValue((v) => v + input);
    }
  });

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1}>
      <Box>
        <Text color="yellow">◉ </Text>
        <Text>Nhập: <Text color="yellow">MÃ_MÓN</Text> <Text color="yellow">SL</Text>  ·  <Text color="yellow">00</Text>/Enter trống = Xong</Text>
      </Box>
      <Text><Text color="cyan">▶</Text> <Text color="cyan" bold>{value}</Text><Text color="gray">{disabled ? ' (đủ 5 món, Enter để xong)' : '_'}</Text></Text>
      {flash ? <Text color="red">◆ {flash}</Text> : null}
    </Box>
  );
}

function App({ serverIp, clientId }) {
  const { exit } = useApp();

  const [state, setState]       = useState(S.CONNECTING);
  const [sessionCode, setSessionCode] = useState('');
  const [menu, setMenu]         = useState([]);
  const [suggest, setSuggest]   = useState([]);
  const [userId, setUserId]     = useState(-1);
  const [isNewUser, setIsNewUser] = useState(true);
  const [orderCount, setOrderCount] = useState(0);
  const [phone, setPhone]       = useState('');
  const [userName, setUserName] = useState('');
  const [order, setOrder]       = useState([]);
  const [invoice, setInvoice]   = useState(null);
  const [lastAck, setLastAck]   = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  const ipc = useMemo(() => createIpc({ serverIp, clientId }), [serverIp, clientId]);

  useEffect(() => {
    const unsubs = [];
    unsubs.push(ipc.on('connected', () => setState(S.WAITING)));
    unsubs.push(ipc.on('session_start', (e) => {
      setSessionCode(e.code || '');
      setState((prev) => (prev === S.CLOSED ? prev : S.PHONE));
    }));
    unsubs.push(ipc.on('menu', (e) => setMenu(e.items || [])));
    unsubs.push(ipc.on('user_ack', (e) => {
      setUserId(e.userId);
      setIsNewUser(!!e.isNew);
      setOrderCount(e.orderCount || 0);
      setUserName(e.name || '');
      if (e.userId === 0) {
        setErrorMsg('SĐT không hợp lệ. Vui lòng thử lại.');
        setState(S.PHONE);
        return;
      }
      // Khach moi (chua co ten) -> chuyen sang NAME_INPUT
      if (e.isNew) {
        setState(S.NAME_INPUT);
      }
      // Khach cu -> cho SUGGEST -> ORDERING (xu ly trong ipc.on('suggest'))
    }));
    unsubs.push(ipc.on('suggest', (e) => {
      setSuggest(e.items || []);
      setState((prev) => (prev === S.LOADING || prev === S.REGISTERING) ? S.ORDERING : prev);
    }));
    unsubs.push(ipc.on('invoice_ready', (e) => {
      setInvoice({
        items: e.items || [],
        subtotal: e.subtotal || 0,
        discount: e.discount || 0,
        total: e.total || 0
      });
      setState(S.CHECKOUT);
    }));
    unsubs.push(ipc.on('order_ack', (e) => { setLastAck(e); setState(S.THANKS); }));
    unsubs.push(ipc.on('session_stop', () => setState(S.CLOSED)));
    unsubs.push(ipc.on('disconnected', () => {
      setState((prev) => (prev === S.CLOSED ? prev : S.ERROR));
      setErrorMsg('Mất kết nối với Server.');
    }));
    unsubs.push(ipc.on('error', (e) => setErrorMsg(e.message || 'lỗi không rõ')));
    unsubs.push(ipc.onExit(() => {
      setState((prev) => (prev === S.CLOSED ? prev : S.ERROR));
    }));
    return () => unsubs.forEach((u) => u && u());
  }, [ipc]);

  useEffect(() => {
    if (state === S.CLOSED || state === S.ERROR) {
      const t = setTimeout(() => { ipc.quit(); exit(); }, 2000);
      return () => clearTimeout(t);
    }
  }, [state, ipc, exit]);

  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) { ipc.quit(); exit(); }
  });

  const onPhoneSubmit = (p) => {
    setPhone(p); setErrorMsg(''); setOrder([]); setUserName('');
    ipc.send({ cmd: 'login', phone: p });
    setState(S.LOADING);
  };

  const onNameSubmit = (name, desc) => {
    setUserName(name);
    ipc.send({ cmd: 'register', phone, name, desc: desc || '' });
    setState(S.REGISTERING);
  };

  const onAddItem = (code, qty) => {
    const m = menu.find((x) => x.code === code);
    if (!m) { setErrorMsg(`Mã món '${code}' không tồn tại`); return; }
    if (order.length >= 5) return;
    setOrder([...order, { code, qty, price: m.price, name: m.name }]);
    setErrorMsg('');
    ipc.send({ cmd: 'add_item', code, qty });
  };

  const onFinish = () => {
    if (order.length === 0) { setErrorMsg('Đơn trống, thêm món trước khi kết thúc.'); return; }
    ipc.send({ cmd: 'finish' });
  };
  const onConfirm = () => { ipc.send({ cmd: 'confirm' }); setState(S.SUBMITTING); };
  const onCancel  = () => {
    ipc.send({ cmd: 'cancel' });
    setOrder([]); setInvoice(null); setState(S.ORDERING);
  };
  const onAfterThanks = () => {
    setOrder([]); setInvoice(null); setLastAck(null); setPhone('');
    setState(S.PHONE);
  };

  // Running subtotal cho hien thi thoi gian thuc
  const runningSubtotal = order.reduce((s, it) => s + it.price * it.qty, 0);

  let body;
  switch (state) {
    case S.CONNECTING:
      body = <Box><Text color="yellow"><Spinner type="bouncingBar" /></Text><Text color="yellow"> ◆ Kết nối {serverIp}:8888...</Text></Box>;
      break;
    case S.WAITING:
      body = <WaitingScreen />;
      break;
    case S.PHONE:
      body = <PhoneInput onSubmit={onPhoneSubmit} errorMsg={errorMsg} />;
      break;
    case S.LOADING:
      body = <Box><Text color="cyan"><Spinner type="dots" /></Text><Text color="cyan"> ▲ Đang tải gợi ý từ LFM...</Text></Box>;
      break;
    case S.NAME_INPUT:
      body = <NameInput phone={phone} onSubmit={onNameSubmit} />;
      break;
    case S.REGISTERING:
      body = <Box><Text color="magenta"><Spinner type="star" /></Text><Text color="magenta"> ★ Đang lưu thông tin...</Text></Box>;
      break;
    case S.ORDERING:
      body = (
        <Box flexDirection="column">
          <Box marginBottom={1}>
            {userName
              ? <Text color="green" bold>◆  Chào mừng trở lại, <Text color="cyan">{userName}</Text>! Bạn đã đặt <Text color="cyan">{orderCount}</Text> đơn.</Text>
              : <Text color="green" bold>★  Chào mừng trở lại! Bạn đã đặt <Text color="cyan">{orderCount}</Text> đơn trước.</Text>
            }
          </Box>
          <Box flexDirection="row">
            <MenuDisplay items={menu} selectedCodes={order.map((o) => o.code)} />
            <Box marginLeft={1}>
              <SuggestPanel items={suggest} />
            </Box>
          </Box>
          <Box marginTop={1}>
            <OrderSummary items={order} max={5} subtotal={runningSubtotal} />
          </Box>
          <Box marginTop={1}>
            <OrderInput onAdd={onAddItem} onFinish={onFinish} disabled={order.length >= 5} />
          </Box>
          {errorMsg ? <Box marginTop={1}><Text color="red">{errorMsg}</Text></Box> : null}
        </Box>
      );
      break;
    case S.CHECKOUT:
      body = <Invoice data={invoice} phone={phone} sessCode={sessionCode}
                      onConfirm={onConfirm} onCancel={onCancel} />;
      break;
    case S.SUBMITTING:
      body = <Box><Text color="magenta"><Spinner type="squareCorners" /></Text><Text color="magenta"> ◉ Đang gửi đơn lên Server...</Text></Box>;
      break;
    case S.THANKS:
      body = <DailySummary ack={lastAck} onNext={onAfterThanks} />;
      break;
    case S.CLOSED:
      body = <Box borderStyle="double" borderColor="magenta" paddingX={2} paddingY={1}>
        <Text color="magenta" bold>Ca làm việc đã kết thúc. Cảm ơn quý khách!</Text>
      </Box>;
      break;
    case S.ERROR:
      body = <Box borderStyle="double" borderColor="red" paddingX={2} paddingY={1}>
        <Text color="red" bold>Lỗi: {errorMsg || 'không xác định'}</Text>
      </Box>;
      break;
    default: body = null;
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      <TitleBar clientId={clientId} sessionCode={sessionCode} phone={phone} state={state} />
      <Box marginTop={1}>{body}</Box>
      <Box marginTop={1}>
        <Text dimColor>(Esc hoặc Ctrl+C để thoát)</Text>
      </Box>
    </Box>
  );
}

const args = process.argv.slice(2);
const serverIp = args[0] || '127.0.0.1';
const clientId = parseInt(args[1] || '1', 10);

render(<App serverIp={serverIp} clientId={clientId} />);
