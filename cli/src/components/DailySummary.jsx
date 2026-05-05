// DailySummary.jsx — Thanks screen sau khi ORDER_ACK
import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';

export default function DailySummary({ ack, onNext }) {
  const [countdown, setCountdown] = useState(3);

  useInput((input, key) => {
    if (key.return || input === ' ') onNext();
  });

  useEffect(() => {
    if (countdown <= 0) { onNext(); return; }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown, onNext]);

  const ok = ack && ack.status === 'OK';
  return (
    <Box flexDirection="column" borderStyle="double" borderColor={ok ? 'green' : 'red'} paddingX={3} paddingY={1}>
      {ok ? (
        <>
          <Box justifyContent="center">
            <Text color="yellow" bold>★  </Text>
            <Text color="green" bold>CẢM ƠN QUÝ KHÁCH!</Text>
            <Text color="yellow" bold>  ★</Text>
          </Box>
          <Box marginTop={1}>
            <Text color="green">◆ </Text>
            <Text>Đơn hàng #<Text color="yellow" bold>{ack?.orderId ?? '?'}</Text> đã được ghi nhận trên hệ thống.</Text>
          </Box>
          <Box>
            <Text color="green">▲ </Text>
            <Text dimColor>Món Anh/Chị sẽ sớm được đơn vị phục vụ.</Text>
          </Box>
          <Box marginTop={1}>
            <Text dimColor>Tiếp tục phục vụ khách mới trong <Text color="cyan">{countdown}s</Text>... (Enter để tiếp ngay)</Text>
          </Box>
        </>
      ) : (
        <>
          <Text color="red" bold>▲ Không thể gửi đơn</Text>
          <Text>◇ Phản hồi Server: {ack?.status || 'không rõ'}</Text>
        </>
      )}
    </Box>
  );
}
