#ifndef APP_ORDER_DTO_H
#define APP_ORDER_DTO_H

#include <cstdint>
#include <string>
#include <vector>

namespace app {

struct ItemAddedRequest {
    int          clientId;
    int64_t      userId;
    std::string  newCode;        // mã món vừa thêm
    std::string  currentCodes;   // các mã hiện có trong đơn (CSV) - dùng để loại khỏi gợi ý
};

struct OrderItemDto {
    std::string code;
    int         qty;
};

struct OrderSubmitRequest {
    int                       clientId;
    int64_t                   userId;
    std::vector<OrderItemDto> items;
    double                    clientTotal;       // chỉ kiểm tra tham khảo, server tính lại
    double                    clientDiscount;
};

struct OrderResponse {
    int64_t      orderId;
    bool         ok;
    std::string  failReason;
};

} // namespace app

#endif
