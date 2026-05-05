// WaitingScreen.jsx — Memphis style + spinner
import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';

export default function WaitingScreen() {
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={3} paddingY={1}>
      <Box>
        <Text color="yellow" bold>◆  NHÀ HÀNG VIỆT PHONG  ◆</Text>
      </Box>
      <Box marginTop={1}>
        <Text color="yellow" bold>
          <Spinner type="bouncingBar" />
          {'  '}CHỜ THU NGÂN MỞ CA{'  '}
          <Spinner type="bouncingBar" />
        </Text>
      </Box>
      <Box marginTop={1}>
        <Text dimColor>▲ Thu ngân nhập MÃ SỐ + Enter tại máy chủ để bắt đầu phục vụ.</Text>
      </Box>
    </Box>
  );
}
