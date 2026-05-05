#ifndef DISPLAY_H
#define DISPLAY_H

#include "order_builder.h"

void showMenu();
void showSuggestions(const char* suggestPayload);
void showOrderStatus(const ClientOrder* o);
void showInvoice(const ClientOrder* o, int clientId, const char* phone,
                 const char* sessCode);
void showUserAck(const char* userAckPayload);

#endif
