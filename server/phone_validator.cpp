#include "phone_validator.h"
#include <cstring>
#include <cctype>

bool isValidPhone(const char* phone) {
    if (!phone) return false;
    int len = (int)strlen(phone);
    if (len != 10) return false;
    if (phone[0] != '0') return false;
    for (int i = 0; i < 10; i++) {
        if (!isdigit((unsigned char)phone[i])) return false;
    }
    return true;
}
