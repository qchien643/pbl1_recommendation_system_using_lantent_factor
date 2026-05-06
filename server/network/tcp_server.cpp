#include "tcp_server.h"
#include "protocol_codec.h"
#include "../../shared/net.h"
#include <cstring>
#include <cstdio>
#include <vector>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socklen_t = int;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #define INVALID_SOCKET (-1)
  #define SOCKET_ERROR (-1)
  #define closesocket close
#endif

namespace app {

struct TcpServer::Slot {
    long long fd     = (long long)INVALID_SOCKET;
    bool      active = false;
    char      recvBuf[4096];
    int       recvLen = 0;
};

TcpServer::TcpServer(int port)
    : port_(port), running_(false), listenFd_((long long)INVALID_SOCKET)
{
    slots_.resize(MAX_SLOTS);
}

TcpServer::~TcpServer() { stop(); }

bool TcpServer::start() {
    listenFd_ = (long long)socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ == (long long)INVALID_SOCKET) return false;

    int yes = 1;
    setsockopt((int)listenFd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port_);

    if (bind((int)listenFd_, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket((int)listenFd_); listenFd_ = (long long)INVALID_SOCKET;
        return false;
    }
    if (listen((int)listenFd_, MAX_SLOTS) == SOCKET_ERROR) {
        closesocket((int)listenFd_); listenFd_ = (long long)INVALID_SOCKET;
        return false;
    }

    running_ = true;
    acceptThread_ = std::thread([this] { acceptLoop(); });
    recvThread_   = std::thread([this] { recvLoop(); });
    return true;
}

void TcpServer::stop() {
    if (!running_.exchange(false)) return;
    if (listenFd_ != (long long)INVALID_SOCKET) {
        closesocket((int)listenFd_);
        listenFd_ = (long long)INVALID_SOCKET;
    }
    if (acceptThread_.joinable()) acceptThread_.join();
    if (recvThread_.joinable())   recvThread_.join();
    std::lock_guard<std::mutex> lk(slotsMutex_);
    for (size_t i = 0; i < slots_.size(); i++) {
        if (slots_[i].active) {
            closesocket((int)slots_[i].fd);
            slots_[i].active = false;
            slots_[i].fd = (long long)INVALID_SOCKET;
        }
    }
}

void TcpServer::acceptLoop() {
    while (running_) {
        sockaddr_in cli;
        socklen_t   len = sizeof(cli);
        long long fd = (long long)accept((int)listenFd_, (sockaddr*)&cli, &len);
        if (fd == (long long)INVALID_SOCKET) {
            if (running_) continue;
            break;
        }
        int slotIdx = -1;
        {
            std::lock_guard<std::mutex> lk(slotsMutex_);
            for (size_t i = 0; i < slots_.size(); i++) {
                if (!slots_[i].active) {
                    slots_[i].active = true;
                    slots_[i].fd = fd;
                    slots_[i].recvLen = 0;
                    slotIdx = (int)i;
                    break;
                }
            }
        }
        if (slotIdx < 0) {
            closesocket((int)fd);
            continue;
        }
        if (onConnect_) onConnect_(slotIdx);
    }
}

void TcpServer::recvLoop() {
    while (running_) {
        fd_set rfds;
        FD_ZERO(&rfds);
        long long maxFd = -1;
        {
            std::lock_guard<std::mutex> lk(slotsMutex_);
            for (size_t i = 0; i < slots_.size(); i++) {
                if (slots_[i].active) {
                    FD_SET((int)slots_[i].fd, &rfds);
                    if (slots_[i].fd > maxFd) maxFd = slots_[i].fd;
                }
            }
        }
        if (maxFd < 0) {
#ifdef _WIN32
            Sleep(20);
#else
            usleep(20000);
#endif
            continue;
        }
        timeval tv; tv.tv_sec = 0; tv.tv_usec = 50000;
        int ready = select((int)maxFd + 1, &rfds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        for (int i = 0; i < (int)slots_.size(); i++) {
            long long fd;
            {
                std::lock_guard<std::mutex> lk(slotsMutex_);
                if (!slots_[i].active) continue;
                fd = slots_[i].fd;
            }
            if (!FD_ISSET((int)fd, &rfds)) continue;

            char tmp[1024];
            int n = recv((int)fd, tmp, sizeof(tmp), 0);
            if (n <= 0) {
                closeSlot(i);
                continue;
            }

            // Parse lines DƯỚI lock; dispatch onLine_ NGOÀI lock
            // (controllers gọi sendTo lấy cùng mutex — phải tránh deadlock).
            std::vector<std::string> lines;
            {
                std::lock_guard<std::mutex> lk(slotsMutex_);
                if (!slots_[i].active) continue;
                int avail = (int)sizeof(slots_[i].recvBuf) - slots_[i].recvLen - 1;
                if (n > avail) n = avail;
                std::memcpy(slots_[i].recvBuf + slots_[i].recvLen, tmp, n);
                slots_[i].recvLen += n;
                slots_[i].recvBuf[slots_[i].recvLen] = '\0';

                char* p = slots_[i].recvBuf;
                while (true) {
                    char* nl = std::strchr(p, '\n');
                    if (!nl) break;
                    *nl = '\0';
                    lines.emplace_back(p);
                    p = nl + 1;
                }
                int remaining = slots_[i].recvLen - (int)(p - slots_[i].recvBuf);
                if (remaining < 0) remaining = 0;
                if (remaining > 0) std::memmove(slots_[i].recvBuf, p, remaining);
                slots_[i].recvLen = remaining;
                slots_[i].recvBuf[remaining] = '\0';
            }
            if (onLine_) for (const auto& ln : lines) onLine_(i, ln);
        }
    }
}

void TcpServer::closeSlot(int idx) {
    long long fd = (long long)INVALID_SOCKET;
    {
        std::lock_guard<std::mutex> lk(slotsMutex_);
        if (!slots_[idx].active) return;
        fd = slots_[idx].fd;
        slots_[idx].active = false;
        slots_[idx].fd = (long long)INVALID_SOCKET;
        slots_[idx].recvLen = 0;
    }
    if (fd != (long long)INVALID_SOCKET) closesocket((int)fd);
    if (onDisconnect_) onDisconnect_(idx);
}

void TcpServer::sendTo(int slot, MsgType type, const std::string& payload) {
    std::string msg = ProtocolCodec::build(type, payload);
    long long fd;
    {
        std::lock_guard<std::mutex> lk(slotsMutex_);
        if (slot < 0 || slot >= (int)slots_.size() || !slots_[slot].active) return;
        fd = slots_[slot].fd;
    }
    send((int)fd, msg.data(), (int)msg.size(), 0);
}

void TcpServer::broadcast(MsgType type, const std::string& payload) {
    std::string msg = ProtocolCodec::build(type, payload);
    std::vector<long long> targets;
    {
        std::lock_guard<std::mutex> lk(slotsMutex_);
        for (size_t i = 0; i < slots_.size(); i++) {
            if (slots_[i].active) targets.push_back(slots_[i].fd);
        }
    }
    for (long long fd : targets) send((int)fd, msg.data(), (int)msg.size(), 0);
}

int TcpServer::clientCount() const {
    int c = 0;
    for (size_t i = 0; i < slots_.size(); i++) if (slots_[i].active) c++;
    return c;
}

} // namespace app
