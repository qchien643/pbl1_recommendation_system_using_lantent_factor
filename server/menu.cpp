#include "menu.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

bool loadMenu(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return false;
    menuCount = 0;
    char line[256];
    while (fgets(line, sizeof(line), f) && menuCount < MAX_MENU) {
        trimNewline(line);
        if (line[0] == '#' || line[0] == '\0') continue;
        char tokens[4][256];
        int n = splitByPipe(line, tokens, 4);
        if (n < 4) continue;
        strncpy(menuCode[menuCount], tokens[0], 3);
        menuCode[menuCount][3] = '\0';
        strncpy(menuName[menuCount], tokens[1], 49);
        menuName[menuCount][49] = '\0';
        menuPrice[menuCount] = (float)atof(tokens[2]);
        menuCategory[menuCount] = tokens[3][0];
        menuCount++;
    }
    fclose(f);
    return menuCount > 0;
}

int findMenuIndex(const char* code) {
    if (!code) return -1;
    for (int i = 0; i < menuCount; i++) {
        if (strcmp(menuCode[i], code) == 0) return i;
    }
    return -1;
}

bool isValidMenuCode(const char* code) {
    if (!code || strlen(code) != 3) return false;
    char p = code[0];
    bool ok = (p == 'P' || p == 'B' || p == 'C' || p == 'G' ||
               p == 'A' || p == 'D' || p == 'T');
    if (!ok) return false;
    if (!isdigit((unsigned char)code[1]) || !isdigit((unsigned char)code[2])) return false;
    return findMenuIndex(code) >= 0;
}

void serializeMenu(char* out, int cap) {
    int o = 0;
    out[0] = '\0';
    for (int i = 0; i < menuCount && o < cap - 100; i++) {
        if (i > 0) o += snprintf(out + o, cap - o, "|");
        o += snprintf(out + o, cap - o, "%s,%s,%.0f",
                      menuCode[i], menuName[i], menuPrice[i]);
    }
}
