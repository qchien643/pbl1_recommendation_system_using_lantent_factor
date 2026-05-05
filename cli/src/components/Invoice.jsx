// Invoice.jsx — Hoa don + Y/N
import React from 'react';
import { Box, Text, useInput } from 'ink';

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}

export default function Invoice({ data, phone, sessCode, onConfirm, onCancel }) {
  useInput((input, key) => {
    if (key.return || input === 'Y' || input === 'y') { onConfirm(); return; }
    if (input === 'N' || input === 'n') { onCancel(); return; }
  });

  if (!data) return <Text dimColor>Đang tính hóa đơn...</Text>;
  const hasDiscount = (data.discount || 0) > 0;

  return (
    <Box flexDirection="column" borderStyle="double" borderColor="cyan" paddingX={2} paddingY={1}>
      <Box justifyContent="center">
        <Text color="yellow">◆ </Text>
        <Text color="cyan" bold>H Ó A   Đ Ơ N</Text>
        <Text color="yellow"> ◆</Text>
      </Box>
      <Box marginTop={1}>
        <Text color="cyan">● </Text>
        <Text>Mã GD: <Text color="yellow">{sessCode}</Text>     <Text color="cyan">● </Text>SĐT: <Text color="yellow">{phone}</Text></Text>
      </Box>
      <Box marginTop={1} flexDirection="row">
        <Box width={4}><Text dimColor underline>STT</Text></Box>
        <Box width={5}><Text dimColor underline>Mã</Text></Box>
        <Box width={22}><Text dimColor underline>Tên món</Text></Box>
        <Box width={5}><Text dimColor underline>SL</Text></Box>
        <Box width={11}><Text dimColor underline>Đơn giá</Text></Box>
        <Box width={13}><Text dimColor underline>T.tiền</Text></Box>
      </Box>
      {data.items.map((it, i) => (
        <Box key={i} flexDirection="row">
          <Box width={4}><Text>{i + 1}</Text></Box>
          <Box width={5}><Text color="yellow">{it.code}</Text></Box>
          <Box width={22}><Text>{it.name}</Text></Box>
          <Box width={5}><Text>x{it.qty}</Text></Box>
          <Box width={11}><Text>{money(it.price)}</Text></Box>
          <Box width={13}><Text>{money(it.subtotal)}</Text></Box>
        </Box>
      ))}
      <Box marginTop={1} flexDirection="column">
        <Box><Text color="cyan">▲ </Text><Text>Tạm tính:   <Text>{money(data.subtotal)}</Text></Text></Box>
        <Box>
          <Text color={hasDiscount ? 'green' : 'gray'}>▲ </Text>
          <Text>Giảm giá:   {hasDiscount
            ? <Text color="green" bold>-{money(data.discount)} (25%, đơn {'>='} 2.000.000đ)</Text>
            : <Text dimColor>0đ</Text>}
          </Text>
        </Box>
        <Box>
          <Text color="yellow">★ </Text>
          <Text bold>TỔNG CỘNG: <Text color="cyan" bold>{money(data.total)}</Text></Text>
        </Box>
      </Box>
      <Box marginTop={1} borderStyle="single" borderColor="yellow" paddingX={1}>
        <Text color="yellow" bold>◉ Xác nhận? </Text>
        <Text color="green" bold>[Y/Enter]</Text>
        <Text> = Gửi Server    </Text>
        <Text color="red" bold>[N]</Text>
        <Text> = Sửa lại</Text>
      </Box>
    </Box>
  );
}
