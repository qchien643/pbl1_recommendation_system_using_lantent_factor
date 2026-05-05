cmake_minimum_required(VERSION 3.15)
project(restaurant_order_app CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/shared ${CMAKE_SOURCE_DIR}/server)

# Winsock2 on Windows
if (WIN32)
    link_libraries(ws2_32)
endif()

# -----------------------------------------------------------------------------
# Phase 1: core (no network)
# -----------------------------------------------------------------------------

set(CORE_SOURCES
    shared/state.cpp
    shared/utils.cpp
    shared/protocol.cpp
    shared/json.cpp
    server/menu.cpp
    server/phone_validator.cpp
    server/user_store.cpp
    server/transaction_store.cpp
    server/order_store.cpp
    server/lfm.cpp
    server/file_manager.cpp
    server/session.cpp
)

add_executable(phase1_test
    tests/phase1_test.cpp
    ${CORE_SOURCES}
)

# -----------------------------------------------------------------------------
# Phase 4: Seed data generator — tao users.dat + lfm_*.dat mau
# -----------------------------------------------------------------------------

add_executable(seed_data
    tools/seed_data.cpp
    ${CORE_SOURCES}
)

# -----------------------------------------------------------------------------
# Phase 2: networking (server + client)
# -----------------------------------------------------------------------------

add_executable(server
    server/main_server.cpp
    server/socket_server.cpp
    shared/net.cpp
    ${CORE_SOURCES}
)

# Client only uses Phase 1 core thanks to shared/state.cpp
add_executable(client
    client/main_client.cpp
    client/socket_client.cpp
    client/order_builder.cpp
    client/input_handler.cpp
    client/display.cpp
    shared/net.cpp
    shared/state.cpp
    shared/utils.cpp
    shared/protocol.cpp
    shared/json.cpp
    server/menu.cpp
    server/phone_validator.cpp
)

# pthread for std::thread on MinGW
if (NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(server PRIVATE Threads::Threads)
endif()
