#include "utils.h"
#include <cstdio>
#include <cstring>
#include <ctime>

void currentTimestamp(char* out, int cap) {
    time_t t = time(NULL);
    struct tm lt;
#ifdef _WIN32
    localtime_s(&lt, &t);
#else
    localtime_r(&t, &lt);
#endif
    snprintf(out, cap, "%04d-%02d-%02d %02d:%02d",
             lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday,
             lt.tm_hour, lt.tm_min);
}

void currentDate(char* out, int cap) {
    time_t t = time(NULL);
    struct tm lt;
#ifdef _WIN32
    localtime_s(&lt, &t);
#else
    localtime_r(&t, &lt);
#endif
    snprintf(out, cap, "%04d-%02d-%02d",
             lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday);
}

void formatMoney(float amount, char* out, int cap) {
    long v = (long)(amount + 0.5f);
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%ld", v);
    int len = (int)strlen(tmp);
    char buf[40];
    int o = 0;
    int first = len % 3;
    if (first == 0 && len > 0) first = 3;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (i - first) % 3 == 0) buf[o++] = '.';
        buf[o++] = tmp[i];
    }
    buf[o] = '\0';
    snprintf(out, cap, "%sd", buf);
}

int splitByPipe(const char* s, char tokens[][256], int maxTokens) {
    int count = 0;
    int o = 0;
    while (*s && count < maxTokens) {
        if (*s == '|') {
            tokens[count][o] = '\0';
            count++;
            o = 0;
            s++;
        } else if (*s == '\n' || *s == '\r') {
            s++;
        } else {
            if (o < 255) tokens[count][o++] = *s;
            s++;
        }
    }
    if (count < maxTokens) {
        tokens[count][o] = '\0';
        count++;
    }
    return count;
}

void trimNewline(char* s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[len - 1] = '\0';
        len--;
    }
}
