#ifndef SESSION_H
#define SESSION_H

bool openSession(const char* code);
bool closeSession(const char* code, const char* dataDir);  // ghi report + luu P/Q + users
bool matchSessionCode(const char* code);
bool isSessionOpen();

#endif
