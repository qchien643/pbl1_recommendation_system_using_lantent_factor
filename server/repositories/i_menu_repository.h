#ifndef APP_I_MENU_REPOSITORY_H
#define APP_I_MENU_REPOSITORY_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace app {

struct MenuItemRecord {
    int64_t     menuIdx = -1;   // == position trong menu table (0..count-1)
    std::string code;
    std::string name;
    double      price = 0.0;
    std::string category;
};

class IMenuRepository {
public:
    virtual ~IMenuRepository() = default;

    virtual void                              clear() = 0;
    virtual int64_t                           save(const MenuItemRecord& item) = 0;  // returns menuIdx assigned

    virtual std::optional<MenuItemRecord>     findByCode(const std::string& code) = 0;
    virtual int64_t                           findIndexByCode(const std::string& code) = 0;  // -1 if not found
    virtual std::optional<MenuItemRecord>     findByIndex(int64_t menuIdx) = 0;
    virtual std::vector<MenuItemRecord>       findAll() = 0;

    virtual int64_t                           count() = 0;
};

} // namespace app

#endif
