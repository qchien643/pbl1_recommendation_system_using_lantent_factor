#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H

bool readLine(char* buf, int cap);
bool readPhone(char* out10);                  // validated 10-digit phone
bool readItemAndQty(char* code, int* qty);    // false khi '00' hoac Enter trong → xong
bool readYesNo();                             // Y/Enter = true, N = false

#endif
