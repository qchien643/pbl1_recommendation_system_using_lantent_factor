#include "socket_client.h"
#include "../shared/net.h"
#include "../shared/protocol.h"
#include <cstdio>
#include <cstring>

static SOCKET sk = INVALID_SOCKET;
static char   recvBuf[2048];
static int    recvLen = 0;

bool cliConnect(const char* ip, int port) {
    sk = socket(AF_INET, SOCK_STREAM, 0);
    if (sk == INVALID_SOCKET) return false;

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    inet_pton(AF_INET, ip, &addr.sin_addr);

    if (connect(sk, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(sk); sk = INVALID_SOCKET;
        return false;
    }
    recvLen = 0;
    return true;
}

void cliDisconnect() {
    if (sk != INVALID_SOCKET) {
        closesocket(sk);
        sk = INVALID_SOCKET;
    }
}

SOCKET cliGetSocket() { return sk; }

int cliRecvOnce(char* buf, int cap) {
    if (sk == INVALID_SOCKET) return -1;
    return recv(sk, buf, cap, 0);
}

bool cliSend(int type, const char* payload) {
    if (sk == INVALID_SOCKET) return false;
    char buf[2048];
    int n = buildMessage((MsgType)type, payload, buf, sizeof(buf));
    return netSendAll(sk, buf, n);
}

bool cliRecvMessage(int* outType, char* payload, int payloadCap) {
    while (true) {
        for (int i = 0; i < recvLen; i++) {
            if (recvBuf[i] == '\n') {
                char line[2048];
                memcpy(line, recvBuf, i);
                line[i] = '\0';
                int rem = recvLen - (i + 1);
                if (rem > 0) memmove(recvBuf, recvBuf + i + 1, rem);
                recvLen = rem;
                ParsedMsg pm;
                if (!parseMessage(line, &pm)) return false;
                *outType = pm.type;
                int plen = (int)strlen(pm.payload);
                if (plen >= payloadCap) plen = payloadCap - 1;
                memcpy(payload, pm.payload, plen);
                payload[plen] = 0;
                return true;
            }
        }
        int space = (int)sizeof(recvBuf) - recvLen;
        if (space <= 0) return false;
        int got = recv(sk, recvBuf + recvLen, space, 0);
        if (got <= 0) return false;
        recvLen += got;
    }
}
