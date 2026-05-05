#ifndef USER_STORE_H
#define USER_STORE_H

int  getOrCreateUser(const char* phone);
int  findUser(const char* phone);
void incrementHistory(int userId, int itemIdx, int qty);
void setUserName(int userId, const char* name, const char* desc);
bool saveUsers(const char* filename);
bool loadUsers(const char* filename);

#endif
