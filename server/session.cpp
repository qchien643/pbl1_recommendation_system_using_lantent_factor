#include "session.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "file_manager.h"
#include "user_store.h"
#include "transaction_store.h"
#include "lfm.h"
#include <cstring>
#include <cstdio>

static bool sessionOpenFlag = false;

bool isSessionOpen() { return sessionOpenFlag; }

bool openSession(const char* code) {
    if (!code || strlen(code) == 0 || strlen(code) >= 10) return false;
    strncpy(sessionCode, code, 9);
    sessionCode[9] = '\0';
    currentTimestamp(sessionStart, 20);
    sessionEnd[0] = '\0';
    sessionOpenFlag = true;
    return true;
}

bool matchSessionCode(const char* code) {
    return code && strcmp(code, sessionCode) == 0;
}

bool closeSession(const char* code, const char* dataDir) {
    if (!matchSessionCode(code)) return false;
    currentTimestamp(sessionEnd, 20);

    char date[16];
    currentDate(date, sizeof(date));

    char reportPath[320], pPath[320], qPath[320], uPath[320], tPath[320];
    snprintf(reportPath, sizeof(reportPath), "%s/reports/report_%s.txt", dataDir, date);
    snprintf(pPath, sizeof(pPath), "%s/lfm_P.dat", dataDir);
    snprintf(qPath, sizeof(qPath), "%s/lfm_Q.dat", dataDir);
    snprintf(uPath, sizeof(uPath), "%s/users.dat",  dataDir);
    snprintf(tPath, sizeof(tPath), "%s/transactions.dat", dataDir);

    writeReport(reportPath);
    lfmSaveModels(pPath, qPath);
    saveUsers(uPath);
    saveTransactions(tPath);
    sessionOpenFlag = false;
    return true;
}
