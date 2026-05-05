// NameInput.jsx — Dang ky ten + mo ta cho khach moi (phong cach Memphis)
import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import Spinner from 'ink-spinner';

const MIN_NAME = 2;
const MAX_NAME = 35;
const MAX_DESC = 79;

// Loc ASCII printable (32..126), chap nhan space va cac ky tu co ban
function sanitize(v) {
  let out = '';
  for (let i = 0; i < v.length; i++) {
    const c = v.charCodeAt(i);
    if (c >= 32 && c <= 126) out += v[i];
  }
  return out;
}

export default function NameInput({ phone, onSubmit }) {
  const [name, setName]   = useState('');
  const [desc, setDesc]   = useState('');
  const [step, setStep]   = useState(0);  // 0 = nhap ten, 1 = nhap mo ta
  const [flash, setFlash] = useState('');

  useInput((input, key) => {
    if (key.return) {
      if (step === 0) {
        const v = sanitize(name).trim();
        if (v.length < MIN_NAME) {
          setFlash(`Tên phải có ít nhất ${MIN_NAME} ký tự ASCII`);
          return;
        }
        setName(v);
        setFlash('');
        setStep(1);
        return;
      }
      // step 1 (desc) — cho phép rỗng = skip
      const d = sanitize(desc).trim();
      onSubmit(name, d);
      return;
    }
    if (key.backspace || key.delete) {
      if (step === 0) setName((v) => v.slice(0, -1));
      else            setDesc((v) => v.slice(0, -1));
      setFlash('');
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      const c = input.charCodeAt(0);
      if (c < 32 || c > 126) {
        setFlash('Chỉ chấp nhận ký tự ASCII (không dấu)');
        return;
      }
      if (step === 0 && name.length < MAX_NAME) setName((v) => v + input);
      if (step === 1 && desc.length < MAX_DESC) setDesc((v) => v + input);
      setFlash('');
    }
  });

  return (
    <Box flexDirection="column" borderStyle="double" borderColor="magenta" paddingX={2} paddingY={1}>
      <Box justifyContent="center">
        <Text color="magenta"><Spinner type="star" /></Text>
        <Text color="magenta" bold>  KHÁCH MỚI — VUI LÒNG CHO BIẾT TÊN  </Text>
        <Text color="magenta"><Spinner type="star" /></Text>
      </Box>

      <Box marginTop={1}>
        <Text color="cyan">◆ </Text>
        <Text>SĐT: <Text color="cyan" bold>{phone}</Text></Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Box>
          <Text color={step === 0 ? 'yellow' : 'gray'} bold>
            {step === 0 ? '▶ ' : '◆ '}Tên của bạn:
          </Text>
        </Box>
        <Box marginLeft={2}>
          <Text color="cyan" bold>{name || ''}</Text>
          {step === 0 ? <Text color="gray">_</Text> : null}
        </Box>
        {step === 0 ? (
          <Box marginLeft={2}>
            <Text dimColor>▲ {MIN_NAME}-{MAX_NAME} ký tự ASCII (không dấu). Enter để tiếp tục.</Text>
          </Box>
        ) : null}
      </Box>

      {step >= 1 ? (
        <Box marginTop={1} flexDirection="column">
          <Box>
            <Text color="yellow" bold>▶ Mô tả ngắn (tùy chọn):</Text>
          </Box>
          <Box marginLeft={2}>
            <Text color="cyan">{desc || ''}</Text>
            <Text color="gray">_</Text>
          </Box>
          <Box marginLeft={2}>
            <Text dimColor>▲ Ví dụ: "Dan van phong, thich Pho Bo". Enter trống để bỏ qua.</Text>
          </Box>
        </Box>
      ) : null}

      {flash ? (
        <Box marginTop={1}>
          <Text color="red">◆ Lỗi: {flash}</Text>
        </Box>
      ) : null}
    </Box>
  );
}
