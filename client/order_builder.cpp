#include "order_builder.h"
#include "../server/menu.h"
#include "../shared/state.h"
#include "../shared/constants.h"
#include <cstdio>
#include <cstring>

void orderInit(ClientOrder* o) {
    memset(o, 0, sizeof(*o));
}

bool orderAddItem(ClientOrder* o, const char* code, int qty) {
    if (o->count >= MAX_ITEMS) return false;
    int idx = findMenuIndex(code);
    if (idx < 0) return false;
    memcpy(o->codes[o->count], menuCode[idx], 4);
    o->qtys[o->count]   = qty;
    o->prices[o->count] = menuPrice[idx];
    strncpy(o->names[o->count], menuName[idx], 49);
    o->names[o->count][49] = '\0';
    o->count++;
    return true;
}

void orderFinalize(ClientOrder* o) {
    o->subtotal = 0;
    for (int i = 0; i < o->count; i++) o->subtotal += o->prices[i] * o->qtys[i];
    o->discount = (o->subtotal >= DISCOUNT_THRESHOLD) ? o->subtotal * DISCOUNT_RATE : 0.0f;
    o->total = o->subtotal - o->discount;
}

void orderSerializeForSubmit(const ClientOrder* o, int clientId, int userId,
                             char* out, int cap) {
    int off = snprintf(out, cap, "%d|%d", clientId, userId);
    for (int i = 0; i < o->count; i++) {
        off += snprintf(out + off, cap - off, "|%s,%d", o->codes[i], o->qtys[i]);
    }
    snprintf(out + off, cap - off, "|%.0f|%.0f", o->total, o->discount);
}

void orderCurrentCodes(const ClientOrder* o, char* out, int cap) {
    int off = 0;
    out[0] = '\0';
    for (int i = 0; i < o->count; i++) {
        if (i > 0) off += snprintf(out + off, cap - off, ",");
        off += snprintf(out + off, cap - off, "%s", o->codes[i]);
    }
}
