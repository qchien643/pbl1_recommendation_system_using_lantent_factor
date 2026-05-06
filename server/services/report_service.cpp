#include "report_service.h"
#include "../../shared/utils.h"
#include "../../shared/constants.h"
#include <cstdio>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace app {

ReportService::ReportService(IUserRepository& userRepo,
                              IMenuRepository& menuRepo,
                              ITransactionRepository& txnRepo,
                              SessionService& sessionService)
    : userRepo_(userRepo), menuRepo_(menuRepo), txnRepo_(txnRepo), sessionService_(sessionService) {}

bool ReportService::writeForCurrentSession(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;

    char date[16];
    currentDate(date, sizeof(date));

    fprintf(f, "==============================================================\n");
    fprintf(f, "   BAO CAO NGAY %s\n", date);
    fprintf(f, "   Ma giao dich : %s\n", sessionService_.currentCode().c_str());
    fprintf(f, "   Ca lam viec  : %s - %s\n",
            sessionService_.startTimestamp().c_str(),
            sessionService_.endTimestamp().c_str());
    fprintf(f, "==============================================================\n\n");

    auto txns = txnRepo_.findBySessionCode(sessionService_.currentCode());
    double totalRev = 0, totalDisc = 0;
    int    discOrders = 0;
    std::unordered_map<std::string, int> itemCount;
    std::set<int64_t> uniqueUsers;

    int orderNo = 0;
    for (const auto& t : txns) {
        orderNo++;
        auto userOpt = userRepo_.findById(t.userId);
        const std::string phone = userOpt.has_value() ? userOpt->phone : "?";
        fprintf(f, "DON #%03d | SDT: %s | %s\n", orderNo, phone.c_str(), t.ts.c_str());
        fprintf(f, "--------------------------------------------------------------\n");

        for (const auto& it : t.items) {
            auto m = menuRepo_.findByCode(it.itemCode);
            const std::string nm = m.has_value() ? m->name : "?";
            char priceBuf[32], subBuf[32];
            formatMoney((float)it.price, priceBuf, sizeof(priceBuf));
            float sub = (float)(it.price * it.qty);
            formatMoney(sub, subBuf, sizeof(subBuf));
            fprintf(f, "  %s  %-22s x%-3d %10s = %12s\n",
                    it.itemCode.c_str(), nm.c_str(), (int)it.qty, priceBuf, subBuf);
            itemCount[it.itemCode] += (int)it.qty;
        }

        char sBuf[32], dBuf[32], tBuf[32];
        formatMoney((float)t.subtotal, sBuf, sizeof(sBuf));
        formatMoney((float)t.discount, dBuf, sizeof(dBuf));
        formatMoney((float)t.total,    tBuf, sizeof(tBuf));
        fprintf(f, "  Tam tinh: %s | Giam: %s | Tong: %s\n\n", sBuf, dBuf, tBuf);

        totalRev  += t.total;
        totalDisc += t.discount;
        if (t.discount > 0) discOrders++;
        uniqueUsers.insert(t.userId);
    }

    char revBuf[32], discBuf[32];
    formatMoney((float)totalRev, revBuf, sizeof(revBuf));
    formatMoney((float)totalDisc, discBuf, sizeof(discBuf));
    fprintf(f, "==============================================================\n");
    fprintf(f, "TONG KET NGAY\n");
    fprintf(f, "  Tong so don        : %d\n", orderNo);
    fprintf(f, "  Tong doanh thu     : %s\n", revBuf);
    fprintf(f, "  Tong giam gia      : %s\n", discBuf);
    fprintf(f, "  Don duoc giam      : %d / %d\n", discOrders, orderNo);
    fprintf(f, "  So SDT khac nhau   : %d\n", (int)uniqueUsers.size());

    // Top-3 món bán chạy
    fprintf(f, "  Mon ban chay       : ");
    std::vector<std::pair<std::string, int>> byCount(itemCount.begin(), itemCount.end());
    std::sort(byCount.begin(), byCount.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < byCount.size() && i < 3; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%s (%d lan)", byCount[i].first.c_str(), byCount[i].second);
    }
    fprintf(f, "\n");
    fprintf(f, "==============================================================\n");
    fprintf(f, "LFM MODEL STATS\n");
    fprintf(f, "  Tong users da hoc  : %lld\n", (long long)userRepo_.count());
    fprintf(f, "  Latent dimensions  : K=%d\n", K);
    fprintf(f, "==============================================================\n");

    fclose(f);
    return true;
}

} // namespace app
