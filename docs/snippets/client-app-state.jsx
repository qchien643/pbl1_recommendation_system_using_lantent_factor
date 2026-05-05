// cli/src/ClientApp.jsx — State machine của UI khách (trích lược)

const S = {
  CONNECTING:  'connecting',
  WAITING:     'waiting',       // chờ thu ngân mở ca
  PHONE:       'phone',         // nhập SĐT
  LOADING:     'loading',       // chờ USER_ACK + SUGGEST
  NAME_INPUT:  'name_input',    // khách mới nhập tên + mô tả
  REGISTERING: 'registering',   // chờ USER_ACK sau USER_REGISTER
  ORDERING:    'ordering',      // menu + suggest + cart + input
  CHECKOUT:    'checkout',      // hóa đơn + Y/N
  SUBMITTING:  'submitting',    // chờ ORDER_ACK
  THANKS:      'thanks',        // cảm ơn quý khách
  CLOSED:      'closed',        // session ended
  ERROR:       'error'
};

function App({ serverIp, clientId }) {
  const [state, setState] = useState(S.CONNECTING);
  const [userName, setUserName] = useState('');

  useEffect(() => {
    ipc.on('session_start', () => setState(S.PHONE));
    ipc.on('user_ack', (e) => {
      setUserName(e.name || '');
      // isNew=true → bắt buộc đăng ký tên trước khi vào ORDERING
      if (e.isNew) setState(S.NAME_INPUT);
    });
    ipc.on('suggest', () => {
      // Khi có suggest mới + đang LOADING/REGISTERING → vào ORDERING
      setState((p) => (p === S.LOADING || p === S.REGISTERING) ? S.ORDERING : p);
    });
    ipc.on('order_ack', () => setState(S.THANKS));
    ipc.on('session_stop', () => setState(S.CLOSED));
  }, [ipc]);

  // Chuyển render theo state — mỗi state có 1 component riêng
  switch (state) {
    case S.WAITING:     return <WaitingScreen />;
    case S.PHONE:       return <PhoneInput onSubmit={onPhoneSubmit} />;
    case S.NAME_INPUT:  return <NameInput phone={phone} onSubmit={onNameSubmit} />;
    case S.ORDERING:    return <OrderingView ... />;
    case S.CHECKOUT:    return <Invoice data={invoice} onConfirm={onConfirm} ... />;
    case S.THANKS:      return <DailySummary ack={lastAck} ... />;
    // ...
  }
}
