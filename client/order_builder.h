#ifndef ORDER_BUILDER_H
#define ORDER_BUILDER_H

#include "../shared/constants.h"

struct ClientOrder {
    char  codes[MAX_ITEMS][4];
    int   qtys[MAX_ITEMS];
    float prices[MAX_ITEMS];
    char  names[MAX_ITEMS][50];
    int   count;
    float subtotal;
    float discount;
    float total;
};

void orderInit(ClientOrder* o);
bool orderAddItem(ClientOrder* o, const char* code, int qty);
void orderFinalize(ClientOrder* o);
void orderSerializeForSubmit(const ClientOrder* o, int clientId, int userId,
                             char* out, int cap);
void orderCurrentCodes(const ClientOrder* o, char* out, int cap);

#endif
