#ifndef APP_MENU_REPOSITORY_H
#define APP_MENU_REPOSITORY_H

#include "i_menu_repository.h"
#include "../../shared/db/database.h"

namespace app {

class MenuRepository : public IMenuRepository {
public:
    explicit MenuRepository(db::Database& db);

    void                              clear() override;
    int64_t                           save(const MenuItemRecord& item) override;

    std::optional<MenuItemRecord>     findByCode(const std::string& code) override;
    int64_t                           findIndexByCode(const std::string& code) override;
    std::optional<MenuItemRecord>     findByIndex(int64_t menuIdx) override;
    std::vector<MenuItemRecord>       findAll() override;

    int64_t                           count() override;

private:
    db::Database& db_;
    db::Table&    table_;

    MenuItemRecord rowToRecord(int64_t idx, const db::Row& r) const;
};

} // namespace app

#endif
