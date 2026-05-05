#ifndef NET_H
#define NET_H

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef int socklen_t;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <errno.h>
  #define SOCKET int
  #define INVALID_SOCKET (-1)
  #define SOCKET_ERROR (-1)
  #define closesocket close
#endif

bool netInit();
void netCleanup();
int  netLastError();

// Send toan bo data (loop cho den khi het hoac loi).
bool netSendAll(SOCKET s, const char* data, int len);

// Drain buffer: scan '\n', goi callback cho moi line hoan chinh, shift phan con lai.
typedef void (*LineHandler)(int slot, const char* line, void* ctx);
int  netDrainBuffer(char* buf, int* len, int slot, LineHandler cb, void* ctx);

#endif
