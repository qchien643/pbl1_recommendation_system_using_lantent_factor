#include "protocol_codec.h"

namespace app {

bool ProtocolCodec::parse(const std::string& raw, ParsedMsg& out) {
    return parseMessage(raw.c_str(), &out);
}

std::string ProtocolCodec::build(MsgType type, const std::string& payload) {
    char buf[2048];
    int n = buildMessage(type, payload.c_str(), buf, sizeof(buf));
    return std::string(buf, n);
}

const char* ProtocolCodec::typeName(MsgType type) {
    return msgTypeName(type);
}

} // namespace app
