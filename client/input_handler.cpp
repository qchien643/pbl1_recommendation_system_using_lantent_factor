#include "input_handler.h"
#include "../shared/utils.h"
#include "../server/phone_validator.h"
#include <cstdio>
#include <cstring>
#include <cctype>

bool readLine(char* buf, int cap) {
    if (!fgets(buf, cap, stdin)) return false;
    trimNewline(buf);
    return true;
}

bool readPhone(char* out10) {
    char line[64];
    for (;;) {
        if (!readLine(line, sizeof(line))) return false;
        if (isValidPhone(line)) {
            memcpy(out10, line, 10);
            out10[10] = '\0';
            return true;
        }
        printf("  Loi: SDT phai 10 chu so va bat dau bang 0. Nhap lai:\n> ");
    }
}

bool readItemAndQty(char* code, int* qty) {
    char line[64];
    if (!readLine(line, sizeof(line))) return false;
    if (line[0] == '\0' || strcmp(line, "00") == 0) {
        code[0] = '\0';
        *qty = 0;
        return false;
    }
    char cBuf[8] = {0};
    int q = 0;
    int n = sscanf(line, "%7s %d", cBuf, &q);
    if (n < 1) return false;
    if (isalpha((unsigned char)cBuf[0])) cBuf[0] = (char)toupper((unsigned char)cBuf[0]);
    int len = (int)strlen(cBuf);
    if (len > 3) len = 3;
    memcpy(code, cBuf, len);
    code[len] = '\0';
    *qty = (n >= 2 && q > 0) ? q : 1;
    return true;
}

bool readYesNo() {
    char line[16];
    if (!readLine(line, sizeof(line))) return false;
    if (line[0] == '\0') return true;
    return (line[0] == 'Y' || line[0] == 'y');
}
