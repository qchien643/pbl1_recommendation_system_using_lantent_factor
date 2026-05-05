#ifndef JSON_H
#define JSON_H

// Lightweight JSON helpers — chi dung cho IPC stdio voi React Ink UI.
// Khong ho tro escaping phuc tap, nested objects, arrays da dang.
// Menu name theo BR16 chi ASCII khong dau → khong can escape.

// Trich xuat gia tri string tai key tu cap cao nhat. Vi du:
//   {"cmd":"login","phone":"0901234567"}
//   jsonGetString(s, "cmd", buf, 32) → "login"
bool jsonGetString(const char* json, const char* key, char* out, int cap);

// Trich xuat so nguyen. Tra ve false neu khong tim thay.
bool jsonGetInt(const char* json, const char* key, int* out);

// Build escaped string (escape \ and "). Hau het truong hop khong can.
void jsonEscape(const char* s, char* out, int cap);

#endif
