#ifndef MENU_H
#define MENU_H

bool loadMenu(const char* filename);
bool isValidMenuCode(const char* code);
int  findMenuIndex(const char* code);  // -1 neu khong tim thay
void serializeMenu(char* out, int cap); // payload cho MENU_DATA

#endif
