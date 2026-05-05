#include "net.h"
#include <cstring>

bool netInit() {
#ifdef _WIN32
    WSADATA w;
    return WSAStartup(MAKEWORD(2, 2), &w) == 0;
#else
    return true;
#endif
}

void netCleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

int netLastError() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

bool netSendAll(SOCKET s, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(s, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

int netDrainBuffer(char* buf, int* len, int slot, LineHandler cb, void* ctx) {
    int processed = 0;
    int i = 0;
    while (i < *len) {
        if (buf[i] == '\n') {
            buf[i] = '\0';
            cb(slot, buf, ctx);
            processed++;
            int rem = *len - (i + 1);
            if (rem > 0) memmove(buf, buf + i + 1, rem);
            *len = rem;
            i = 0;
        } else {
            i++;
        }
    }
    return processed;
}
