#include "menu.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

// Menu được lưu trữ trong db::Table "menu" (HashIndex trên cột "code" → O(1) lookup).
// Parallel arrays (menuCode[], menuName[], menuPrice[], menuCategory[]) là VIEW CACHE
// sync với table — caller code cũ vẫn dùng được, đồng thời findMenuIndex() đi qua HashIndex.

bool loadMenu(const char* filename) {
    db::initRestaurantSchema();
    db::Table& menuT = db::Database::instance().table(db::tbl::MENU);
    menuT.clear();

    FILE* f = fopen(filename, "r");
    if (!f) return false;
    menuCount = 0;
    char line[256];
    while (fgets(line, sizeof(line), f) && menuCount < MAX_MENU) {
        trimNewline(line);
        if (line[0] == '#' || line[0] == '\0') continue;
        char tokens[4][256];
        int n = splitByPipe(line, tokens, 4);
        if (n < 4) continue;

        // Insert vào menu table
        db::Row row(&menuT.schema());
        row.set(db::col::CODE,     std::string(tokens[0]));
        row.set(db::col::NAME,     std::string(tokens[1]));
        row.set(db::col::PRICE,    (double)atof(tokens[2]));
        row.set(db::col::CATEGORY, std::string(1, tokens[3][0]));
        try {
            menuT.insert(std::move(row));
        } catch (...) {
            continue;  // duplicate code → bỏ qua
        }

        // Cache view
        strncpy(menuCode[menuCount], tokens[0], 3);
        menuCode[menuCount][3] = '\0';
        strncpy(menuName[menuCount], tokens[1], 49);
        menuName[menuCount][49] = '\0';
        menuPrice[menuCount] = (float)atof(tokens[2]);
        menuCategory[menuCount] = tokens[3][0];
        menuCount++;
    }
    fclose(f);
    return menuCount > 0;
}

int findMenuIndex(const char* code) {
    if (!code) return -1;
    db::Database& d = db::Database::instance();
    if (!d.has(db::tbl::MENU)) return -1;
    db::RowId rid = d.table(db::tbl::MENU).findOne(db::col::CODE, db::Value(std::string(code)));
    return rid >= 0 ? (int)rid : -1;
}

bool isValidMenuCode(const char* code) {
    if (!code || strlen(code) != 3) return false;
    char p = code[0];
    bool ok = (p == 'P' || p == 'B' || p == 'C' || p == 'G' ||
               p == 'A' || p == 'D' || p == 'T');
    if (!ok) return false;
    if (!isdigit((unsigned char)code[1]) || !isdigit((unsigned char)code[2])) return false;
    return findMenuIndex(code) >= 0;
}

void serializeMenu(char* out, int cap) {
    int o = 0;
    out[0] = '\0';
    for (int i = 0; i < menuCount && o < cap - 100; i++) {
        if (i > 0) o += snprintf(out + o, cap - o, "|");
        o += snprintf(out + o, cap - o, "%s,%s,%.0f",
                      menuCode[i], menuName[i], menuPrice[i]);
    }
}
