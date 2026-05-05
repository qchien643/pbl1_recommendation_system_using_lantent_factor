// PhoneInput.jsx — Nhap SDT 10 chu so bat dau '0', phong cach Memphis
import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import Spinner from 'ink-spinner';

function isValid(p) { return /^0\d{9}$/.test(p); }

export default function PhoneInput({ onSubmit, errorMsg }) {
  const [value, setValue] = useState('');
  const [flash, setFlash] = useState('');

  useInput((input, key) => {
    if (key.return) {
      if (!isValid(value)) { setFlash('SĐT phải 10 chữ số và bắt đầu bằng 0'); return; }
      setFlash(''); onSubmit(value); return;
    }
    if (key.backspace || key.delete) { setValue((v) => v.slice(0, -1)); setFlash(''); return; }
    if (input && /^\d$/.test(input) && value.length < 10) {
      setValue((v) => v + input); setFlash('');
    }
  });

  const slots = Array.from({ length: 10 }, (_, i) => value[i] || '_');

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="blue" paddingX={2} paddingY={1}>
      <Box>
        <Text color="blue"><Spinner type="triangle" /></Text>
        <Text bold color="blue">  NHẬP SỐ ĐIỆN THOẠI để nhận gợi ý cá nhân hóa  </Text>
        <Text color="blue"><Spinner type="triangle" /></Text>
      </Box>
      <Box marginTop={1}>
        <Text color="cyan" bold>◉ SĐT: </Text>
        {slots.map((ch, i) => (
          <Text key={i} color={ch === '_' ? 'gray' : 'cyan'} bold>
            {'['}{ch}{'] '}
          </Text>
        ))}
      </Box>
      <Box marginTop={1}>
        <Text color="blue">▲ </Text>
        <Text dimColor>10 chữ số, bắt đầu bằng 0. Bấm Enter để xác nhận.</Text>
      </Box>
      {flash ? <Text color="red">◆ Lỗi: {flash}</Text> : null}
      {errorMsg ? <Text color="red">◆ {errorMsg}</Text> : null}
    </Box>
  );
}
