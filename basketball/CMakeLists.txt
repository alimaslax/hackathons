cmake_minimum_required(VERSION 3.10)
project(BasketballDribble)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Homebrew's include directory
execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find SFML
set(SFML_DIR "${HOMEBREW_PREFIX}/lib/cmake/SFML")
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add the executable
add_executable(BasketballDribble basketball_dribble.cpp)

# Link SFML libraries
target_link_libraries(BasketballDribble sfml-graphics sfml-window sfml-system)

# Include SFML headers
target_include_directories(BasketballDribble PRIVATE "${HOMEBREW_PREFIX}/include")