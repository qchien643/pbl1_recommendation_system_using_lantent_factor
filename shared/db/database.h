#ifndef DB_DATABASE_H
#define DB_DATABASE_H

#include "table.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace db {

// Database = registry các Table. Singleton tiện cho legacy code dùng global accessor;
// vẫn cho phép tạo instance riêng (test).
class Database {
public:
    Database()  = default;
    ~Database() = default;

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    static Database& instance();

    // Tạo bảng mới — schema phải set xong trước khi gọi.
    Table& createTable(const std::string& name, Schema schema);

    Table&       table(const std::string& name);
    const Table& table(const std::string& name) const;
    bool         has(const std::string& name) const;

    // Load tất cả bảng từ thư mục — file path = dataDir/<name>.tbl.
    // Bảng nào không có file → giữ rỗng (mới khởi tạo).
    bool openAll(const std::string& dataDir);

    // Lưu tất cả bảng xuống thư mục.
    bool saveAll(const std::string& dataDir);

    // Iterate tên bảng
    std::vector<std::string> tableNames() const;

private:
    std::unordered_map<std::string, std::unique_ptr<Table>> tables_;
};

} // namespace db

#endif
