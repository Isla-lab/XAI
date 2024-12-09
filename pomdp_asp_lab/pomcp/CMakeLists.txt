cmake_minimum_required(VERSION 3.9)
project(pomcp CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Boost 1.33 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable(pomcp
    ./src/main.cpp
    ./src/beliefstate.cpp
    ./src/coord.cpp
    ./src/experiment.cpp
    ./src/mcts.cpp
    ./src/network.cpp
    ./src/node.cpp
    ./src/rocksample.cpp
    ./src/simulator.cpp
    ./src/testsimulator.cpp
    ./src/utils.cpp
    ./src/xes_logger.cpp
    ./src/node_lazy.cpp
    ./src/simulator_lazy.cpp
    ./src/mcts_lazy.cpp
    ./src/experiment_lazy.cpp
    )
target_include_directories(pomcp PUBLIC ./src)
target_link_libraries(pomcp LINK_PUBLIC ${Boost_LIBRARIES})
