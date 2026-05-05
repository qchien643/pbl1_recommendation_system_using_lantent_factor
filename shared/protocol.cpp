#include "protocol.h"
#include <cstring>
#include <cstdio>

static const char* NAMES[] = {
    "UNKNOWN",
    "START", "STOP", "MENU_DATA", "USER_ACK",
    "SUGGEST", "ORDER_ACK",
    "USER_LOGIN", "ITEM_ADDED", "ORDER_SUBMIT", "HEARTBEAT",
    "USER_REGISTER"
};

const char* msgTypeName(MsgType t) {
    if (t < 0 || t > MSG_USER_REGISTER) return "UNKNOWN";
    return NAMES[t];
}

MsgType msgTypeFromName(const char* name) {
    for (int i = 1; i <= MSG_USER_REGISTER; i++) {
        if (strcmp(name, NAMES[i]) == 0) return (MsgType)i;
    }
    return MSG_UNKNOWN;
}

bool parseMessage(const char* raw, ParsedMsg* out) {
    if (!raw || !out) return false;
    const char* bar = strchr(raw, '|');
    char typeName[32] = {0};

    if (bar) {
        int n = (int)(bar - raw);
        if (n >= 32) return false;
        memcpy(typeName, raw, n);
        typeName[n] = '\0';
        const char* payload = bar + 1;
        int plen = (int)strlen(payload);
        while (plen > 0 && (payload[plen - 1] == '\n' || payload[plen - 1] == '\r')) plen--;
        if (plen >= (int)sizeof(out->payload)) plen = (int)sizeof(out->payload) - 1;
        memcpy(out->payload, payload, plen);
        out->payload[plen] = '\0';
    } else {
        int n = (int)strlen(raw);
        while (n > 0 && (raw[n - 1] == '\n' || raw[n - 1] == '\r')) n--;
        if (n >= 32) return false;
        memcpy(typeName, raw, n);
        typeName[n] = '\0';
        out->payload[0] = '\0';
    }

    out->type = msgTypeFromName(typeName);
    return out->type != MSG_UNKNOWN;
}

int buildMessage(MsgType type, const char* payload, char* out, int cap) {
    return snprintf(out, cap, "%s|%s\n", msgTypeName(type), payload ? payload : "");
}
