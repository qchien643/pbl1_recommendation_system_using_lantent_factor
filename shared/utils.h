#ifndef UTILS_H
#define UTILS_H

// Format thoi gian va tien
void currentTimestamp(char* out, int cap);   // "2026-04-23 10:35"
void currentDate(char* out, int cap);        // "2026-04-23"
void formatMoney(float amount, char* out, int cap); // "160.000d"

// Chuoi
int  splitByPipe(const char* s, char tokens[][256], int maxTokens);
void trimNewline(char* s);

#endif
