// SuggestPanel.jsx — Top-3 LFM, phong cach Memphis, chi hien code + ten mon
import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';

const RANK = [
  { icon: '★', color: 'yellow' },
  { icon: '◆', color: 'cyan' },
  { icon: '▲', color: 'green' },
];

export default function SuggestPanel({ items }) {
  if (!items || items.length === 0) {
    return (
      <Box borderStyle="round" borderColor="green" paddingX={1} width={32}>
        <Text dimColor>◇  (Chưa có gợi ý từ LFM)</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="green" paddingX={1} width={32}>
      <Box>
        <Text color="green"><Spinner type="star" /></Text>
        <Text color="green" bold>  GỢI Ý CHO BẠN</Text>
      </Box>
      {items.map((it, i) => {
        const r = RANK[i] || RANK[2];
        return (
          <Box key={i} flexDirection="row" marginTop={i === 0 ? 1 : 0}>
            <Box width={3}><Text color={r.color} bold>{r.icon}</Text></Box>
            <Box width={5}><Text color="yellow" bold>{it.code}</Text></Box>
            <Box><Text>{it.name || ''}</Text></Box>
          </Box>
        );
      })}
    </Box>
  );
}
