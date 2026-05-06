#ifndef APP_PROTOCOL_CODEC_H
#define APP_PROTOCOL_CODEC_H

#include "../../shared/protocol.h"
#include <string>

namespace app {

// Wrapper OOP cho protocol parser/builder cũ (shared/protocol.h).
class ProtocolCodec {
public:
    static bool         parse(const std::string& raw, ParsedMsg& out);
    static std::string  build(MsgType type, const std::string& payload);
    static const char*  typeName(MsgType type);
};

} // namespace app

#endif
