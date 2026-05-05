// MenuDisplay.jsx — phong cach Memphis, moi category co icon rieng
import React from 'react';
import { Box, Text } from 'ink';

function money(n) {
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.') + 'd';
}

const CAT_COLOR = {
  P: 'red', B: 'magenta', C: 'yellow',
  G: 'green', A: 'green', D: 'cyan', T: 'magentaBright'
};

// Memphis icon cho tung nhom mon
const CAT_ICON = {
  P: '◆',  // Pho
  B: '▲',  // Bun
  C: '■',  // Com
  G: '●',  // Goi
  A: '★',  // An vat
  D: '◉',  // Do uong
  T: '◈',  // Trang mieng
};

export default function MenuDisplay({ items, selectedCodes = [] }) {
  const sel = new Set(selectedCodes);
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={44}>
      <Text bold>◆ THỰC ĐƠN ({items.length} món) ◆</Text>
      <Box flexDirection="row" marginBottom={0}>
        <Box width={3}><Text dimColor underline> </Text></Box>
        <Box width={6}><Text dimColor underline>MÃ</Text></Box>
        <Box width={22}><Text dimColor underline>TÊN MÓN</Text></Box>
        <Box width={11}><Text dimColor underline>GIÁ</Text></Box>
      </Box>
      {items.map((it) => {
        const chosen = sel.has(it.code);
        const color = chosen ? 'gray' : (CAT_COLOR[it.code?.[0]] || 'white');
        const icon = CAT_ICON[it.code?.[0]] || '◇';
        return (
          <Box key={it.code} flexDirection="row">
            <Box width={3}>
              <Text color={chosen ? 'gray' : color} bold>{chosen ? '✓' : icon}</Text>
            </Box>
            <Box width={6}>
              <Text color={color} bold={!chosen}>{it.code}</Text>
            </Box>
            <Box width={22}>
              <Text color={chosen ? 'gray' : 'white'} strikethrough={chosen}>{it.name}</Text>
            </Box>
            <Box width={11}>
              <Text color={chosen ? 'gray' : 'white'}>{money(it.price)}</Text>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}
