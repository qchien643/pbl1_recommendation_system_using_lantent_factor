#include "json.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

bool jsonGetString(const char* json, const char* key, char* out, int cap) {
    if (!json || !key || !out || cap <= 0) return false;
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    if (*p != ':') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return false;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < cap - 1) {
        if (*p == '\\' && *(p + 1)) { out[i++] = *(p + 1); p += 2; }
        else out[i++] = *p++;
    }
    out[i] = '\0';
    return true;
}

bool jsonGetInt(const char* json, const char* key, int* out) {
    if (!json || !key || !out) return false;
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    if (*p != ':') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    *out = atoi(p);
    return true;
}

void jsonEscape(const char* s, char* out, int cap) {
    int o = 0;
    for (int i = 0; s[i] && o < cap - 2; i++) {
        char c = s[i];
        if (c == '"' || c == '\\') {
            if (o >= cap - 3) break;
            out[o++] = '\\';
            out[o++] = c;
        } else if (c == '\n') {
            if (o >= cap - 3) break;
            out[o++] = '\\';
            out[o++] = 'n';
        } else {
            out[o++] = c;
        }
    }
    out[o] = '\0';
}
