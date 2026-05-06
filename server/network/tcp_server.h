#ifndef APP_TCP_SERVER_H
#define APP_TCP_SERVER_H

#include "../../shared/protocol.h"
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

namespace app {

// TcpServer — encapsulate TCP listener, client slot management, send/recv.
// Dispatch raw lines tới callback (thường là MessageRouter::route).
class TcpServer {
public:
    using LineHandler       = std::function<void(int slot, const std::string& line)>;
    using ConnectHandler    = std::function<void(int slot)>;
    using DisconnectHandler = std::function<void(int slot)>;

    explicit TcpServer(int port);
    ~TcpServer();

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    bool        start();
    void        stop();
    bool        isRunning() const { return running_.load(); }

    void        setOnLine(LineHandler cb)             { onLine_ = std::move(cb); }
    void        setOnConnect(ConnectHandler cb)       { onConnect_ = std::move(cb); }
    void        setOnDisconnect(DisconnectHandler cb) { onDisconnect_ = std::move(cb); }

    void        sendTo(int slot, MsgType type, const std::string& payload);
    void        broadcast(MsgType type, const std::string& payload);

    int         clientCount() const;
    int         port() const { return port_; }

private:
    struct Slot;
    static constexpr int MAX_SLOTS = 20;

    int                  port_;
    std::atomic<bool>    running_;
    std::thread          acceptThread_;
    std::thread          recvThread_;
    mutable std::mutex   slotsMutex_;
    std::vector<Slot>    slots_;
    long long            listenFd_;

    LineHandler          onLine_;
    ConnectHandler       onConnect_;
    DisconnectHandler    onDisconnect_;

    void                 acceptLoop();
    void                 recvLoop();
    void                 closeSlot(int idx);
};

} // namespace app

#endif
