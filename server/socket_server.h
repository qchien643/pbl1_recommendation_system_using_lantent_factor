#ifndef SOCKET_SERVER_H
#define SOCKET_SERVER_H

// Callback hooks cho main_server inject handler (vd emit JSON event cho UI).
// Neu khong set thi socket_server tu printf text logs.
struct SrvHooks {
    void (*onReady)(int port);
    void (*onClientJoined)(int slot);
    void (*onClientLeft)(int slot);
    void (*onUserLogin)(int slot, const char* phone, int userId, bool isNew, int orderCount);
    void (*onItemAdded)(int slot, int userId, const char* code, int excludedCount);
    void (*onOrderSubmitted)(int slot, int userId, int orderId, int itemCount, float total, float discount);
    void (*onHeartbeat)(int slot);
    void (*onUserRegister)(int slot, const char* phone, int userId, const char* name);
};

void srvSetHooks(const SrvHooks* hooks);  // NULL de reset ve text mode mac dinh

bool srvStart(int port);
void srvStop();
void srvPoll(int timeoutMs);
int  srvClientCount();

void srvBroadcastStart(const char* sessionCode, const char* dateTime);
void srvBroadcastStop(const char* dateTime);
void srvBroadcastMenu();

#endif
