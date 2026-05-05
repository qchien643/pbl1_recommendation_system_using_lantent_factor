#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include "../shared/net.h"

bool cliConnect(const char* ip, int port);
void cliDisconnect();
bool cliSend(int msgType, const char* payload);
bool cliRecvMessage(int* outType, char* payload, int payloadCap);  // blocking

// Phase 3 (JSON IPC): raw access cho select() loop
SOCKET cliGetSocket();
int    cliRecvOnce(char* buf, int cap);  // 1 lan recv, <=0 = disconnect

#endif
