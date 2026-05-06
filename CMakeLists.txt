cmake_minimum_required(VERSION 3.15)
project(restaurant_order_app CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/shared ${CMAKE_SOURCE_DIR}/server)

if (WIN32)
    link_libraries(ws2_32)
endif()

# =============================================================================
# Layer 1: Mini-DBMS — Tables + B-tree/Hash/Fenwick indexes
# =============================================================================

set(DB_SOURCES
    shared/db/value.cpp
    shared/db/schema.cpp
    shared/db/codec.cpp
    shared/db/row.cpp
    shared/db/hash_index.cpp
    shared/db/btree_index.cpp
    shared/db/fenwick.cpp
    shared/db/table.cpp
    shared/db/database.cpp
    shared/db/db_schema.cpp
)

# =============================================================================
# Layer 2: Repositories — abstract qua Database, mỗi entity 1 repo
# =============================================================================

set(REPOSITORY_SOURCES
    server/repositories/user_repository.cpp
    server/repositories/menu_repository.cpp
    server/repositories/transaction_repository.cpp
    server/repositories/lfm_repository.cpp
    server/repositories/session_repository.cpp
)

# =============================================================================
# Layer 3: Services — business logic (Auth, Menu, Order, Lfm, Session, Report)
# =============================================================================

set(SERVICE_SOURCES
    server/services/menu_service.cpp
    server/services/auth_service.cpp
    server/services/lfm_service.cpp
    server/services/order_service.cpp
    server/services/session_service.cpp
    server/services/report_service.cpp
)

# =============================================================================
# Layer 4: Controllers + Network + Infrastructure (DI, listeners, lifecycle)
# =============================================================================

set(CONTROLLER_SOURCES
    server/controllers/auth_controller.cpp
    server/controllers/order_controller.cpp
    server/controllers/heartbeat_controller.cpp
)

set(NETWORK_SOURCES
    server/network/protocol_codec.cpp
    server/network/tcp_server.cpp
    server/network/message_router.cpp
)

set(INFRA_SOURCES
    server/infra/text_log_listener.cpp
    server/infra/json_event_listener.cpp
    server/infra/session_lifecycle.cpp
    server/infra/application_context.cpp
)

# =============================================================================
# Shared sources (used by all targets)
# =============================================================================

set(SHARED_SOURCES
    shared/state.cpp
    shared/utils.cpp
    shared/protocol.cpp
    shared/json.cpp
)

# =============================================================================
# Server: full layered stack
# =============================================================================

add_executable(server
    server/main_server.cpp
    shared/net.cpp
    ${SHARED_SOURCES}
    ${DB_SOURCES}
    ${REPOSITORY_SOURCES}
    ${SERVICE_SOURCES}
    ${CONTROLLER_SOURCES}
    ${NETWORK_SOURCES}
    ${INFRA_SOURCES}
    server/menu.cpp                 # legacy free-fn shims (used by tools)
    server/phone_validator.cpp
)

# =============================================================================
# Client: minimal subset
# =============================================================================

add_executable(client
    client/main_client.cpp
    client/socket_client.cpp
    client/order_builder.cpp
    client/input_handler.cpp
    client/display.cpp
    shared/net.cpp
    ${SHARED_SOURCES}
    ${DB_SOURCES}
    server/menu.cpp
    server/phone_validator.cpp
)

# =============================================================================
# Smoke test (exercises core db + repos + services)
# =============================================================================

add_executable(phase1_test
    tests/phase1_test.cpp
    ${SHARED_SOURCES}
    ${DB_SOURCES}
    ${REPOSITORY_SOURCES}
    ${SERVICE_SOURCES}
    server/menu.cpp
    server/phone_validator.cpp
)

# =============================================================================
# Seed data generator
# =============================================================================

add_executable(seed_data
    tools/seed_data.cpp
    ${SHARED_SOURCES}
    ${DB_SOURCES}
    ${REPOSITORY_SOURCES}
    ${SERVICE_SOURCES}
    server/menu.cpp
    server/phone_validator.cpp
)

# =============================================================================
# Migration tool: legacy .dat → .tbl
# =============================================================================

add_executable(migrate_legacy
    tools/migrate_legacy.cpp
    ${SHARED_SOURCES}
    ${DB_SOURCES}
    ${REPOSITORY_SOURCES}
    ${SERVICE_SOURCES}
    server/menu.cpp
    server/phone_validator.cpp
)

# pthread for std::thread on MinGW
if (NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(server PRIVATE Threads::Threads)
endif()
