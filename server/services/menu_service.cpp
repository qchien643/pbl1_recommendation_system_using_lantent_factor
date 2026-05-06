#include "menu_service.h"
#include "../../shared/utils.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

namespace app {

MenuService::MenuService(IMenuRepository& repo) : repo_(repo) {}

bool MenuService::loadFromFile(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    repo_.clear();

    char line[256];
    int loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        trimNewline(line);
        if (line[0] == '#' || line[0] == '\0') continue;
        char tokens[4][256];
        int n = splitByPipe(line, tokens, 4);
        if (n < 4) continue;
        MenuItemRecord m;
        m.code     = tokens[0];
        m.name     = tokens[1];
        m.price    = atof(tokens[2]);
        m.category = std::string(1, tokens[3][0]);
        if (repo_.save(m) >= 0) loaded++;
    }
    fclose(f);
    return loaded > 0;
}

bool MenuService::isValidCode(const std::string& code) {
    if (code.size() != 3) return false;
    char p = code[0];
    bool prefixOk = (p == 'P' || p == 'B' || p == 'C' || p == 'G' ||
                     p == 'A' || p == 'D' || p == 'T');
    if (!prefixOk) return false;
    if (!isdigit((unsigned char)code[1]) || !isdigit((unsigned char)code[2])) return false;
    return repo_.findIndexByCode(code) >= 0;
}

std::string MenuService::serializeForBroadcast() {
    auto items = repo_.findAll();
    std::string out;
    char buf[128];
    for (size_t i = 0; i < items.size(); i++) {
        if (i > 0) out += "|";
        snprintf(buf, sizeof(buf), "%s,%s,%.0f", items[i].code.c_str(), items[i].name.c_str(), items[i].price);
        out += buf;
    }
    return out;
}

int64_t MenuService::findIndexByCode(const std::string& code) {
    return repo_.findIndexByCode(code);
}
std::optional<MenuItemRecord> MenuService::findByCode(const std::string& code) {
    return repo_.findByCode(code);
}
std::optional<MenuItemRecord> MenuService::findByIndex(int64_t idx) {
    return repo_.findByIndex(idx);
}
std::vector<MenuItemRecord> MenuService::findAll() { return repo_.findAll(); }
int64_t MenuService::count() { return repo_.count(); }

} // namespace app
