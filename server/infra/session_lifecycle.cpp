#include "session_lifecycle.h"
#include "../../shared/utils.h"
#include <cstdio>

namespace app {

SessionLifecycle::SessionLifecycle(SessionService& session,
                                    MenuService& menu,
                                    LfmService& lfm,
                                    ReportService& report,
                                    TcpServer& tcp,
                                    db::Database& db,
                                    IServerEventListener& events)
    : session_(session), menu_(menu), lfm_(lfm), report_(report),
      tcp_(tcp), db_(db), events_(events) {}

bool SessionLifecycle::start(const std::string& code) {
    if (!session_.open(code)) return false;

    events_.onSessionOpened(session_.currentCode(), session_.startTimestamp());

    // Broadcast START + MENU_DATA
    char buf[64]; snprintf(buf, sizeof(buf), "%s|%s",
                            session_.currentCode().c_str(),
                            session_.startTimestamp().c_str());
    tcp_.broadcast(MSG_START, buf);

    std::string menuPayload = menu_.serializeForBroadcast();
    tcp_.broadcast(MSG_MENU_DATA, menuPayload);
    return true;
}

bool SessionLifecycle::stop(const std::string& code) {
    if (!session_.close(code)) return false;

    char date[16];
    currentDate(date, sizeof(date));
    char path[320];
    snprintf(path, sizeof(path), "data/reports/report_%s.txt", date);

    report_.writeForCurrentSession(path);
    lfm_.saveToRepository();
    db_.saveAll("data");

    tcp_.broadcast(MSG_STOP, session_.endTimestamp());
    events_.onSessionClosed(session_.endTimestamp());
    return true;
}

} // namespace app
