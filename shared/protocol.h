#ifndef PROTOCOL_H
#define PROTOCOL_H

enum MsgType {
    MSG_UNKNOWN = 0,
    // Server -> Client
    MSG_START,
    MSG_STOP,
    MSG_MENU_DATA,
    MSG_USER_ACK,
    MSG_SUGGEST,
    MSG_ORDER_ACK,
    // Client -> Server
    MSG_USER_LOGIN,
    MSG_ITEM_ADDED,
    MSG_ORDER_SUBMIT,
    MSG_HEARTBEAT,
    MSG_USER_REGISTER     // khach moi gui ten + mo ta
};

struct ParsedMsg {
    MsgType type;
    char    payload[1024];
};

bool        parseMessage(const char* raw, ParsedMsg* out);
int         buildMessage(MsgType type, const char* payload, char* out, int cap);
const char* msgTypeName(MsgType t);
MsgType     msgTypeFromName(const char* name);

#endif
