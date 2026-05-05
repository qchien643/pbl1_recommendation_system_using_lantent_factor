// OrderSummary.jsx — Hien thi don hien tai
import React from 'react';
import { Box, Text } from 'ink';

function money(n) {
  return Math.round(n || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}

export default function OrderSummary({ items = [], max = 5, subtotal = 0 }) {
  const remaining = Math.max(0, max - items.length);
  const willDiscount = subtotal >= 2000000;

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={willDiscount ? 'green' : 'magenta'} paddingX={1}>
      <Box flexDirection="row">
        <Text color="magenta" bold>◆ ĐƠN HIỆN TẠI: </Text>
        {items.length === 0
          ? <Text dimColor>◇ (chưa có món nào)</Text>
          : items.map((it, i) => (
              <Text key={i}> <Text color="yellow">▪ [{it.code} x{it.qty}]</Text></Text>
            ))
        }
      </Box>
      <Box flexDirection="row" marginTop={items.length > 0 ? 0 : 0}>
        <Text color="cyan">● </Text>
        <Text>Số món: <Text color="cyan">{items.length}/{max}</Text></Text>
        <Text>   <Text color="cyan">● </Text>Còn lại: <Text color="cyan">{remaining}</Text></Text>
        <Text>   <Text color={willDiscount ? 'green' : 'white'}>◆ </Text>Tạm tính: <Text color={willDiscount ? 'green' : 'white'} bold>{money(subtotal)}</Text></Text>
        {willDiscount ? <Text color="green" bold>   ★ sẽ được giảm 25%!</Text> : null}
      </Box>
    </Box>
  );
}
