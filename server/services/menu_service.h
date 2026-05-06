#ifndef APP_MENU_SERVICE_H
#define APP_MENU_SERVICE_H

#include "../repositories/i_menu_repository.h"
#include <string>
#include <vector>

namespace app {

class MenuService {
public:
    explicit MenuService(IMenuRepository& repo);

    // Đọc menu.txt format CODE|NAME|PRICE|CATEGORY (#-comment skip).
    // Reset menu repo + insert từng dòng. Return true nếu load >0 món.
    bool loadFromFile(const std::string& path);

    // Validate code: 3 chars, prefix [PBCGADT], 2 digit + tồn tại trong repo.
    bool isValidCode(const std::string& code);

    // Build payload "P01,Pho Bo,65000|B01,Bun Bo,60000|..." cho MENU_DATA broadcast.
    std::string serializeForBroadcast();

    int64_t                               findIndexByCode(const std::string& code);
    std::optional<MenuItemRecord>         findByCode(const std::string& code);
    std::optional<MenuItemRecord>         findByIndex(int64_t menuIdx);
    std::vector<MenuItemRecord>           findAll();
    int64_t                               count();

private:
    IMenuRepository& repo_;
};

} // namespace app

#endif
