#ifndef APP_MENU_DTO_H
#define APP_MENU_DTO_H

#include <cstdint>
#include <string>
#include <vector>

namespace app {

struct SuggestionItem {
    std::string  code;
    float        score;
};

struct SuggestionsResponse {
    int64_t                      userId;
    std::vector<SuggestionItem>  items;
};

} // namespace app

#endif
