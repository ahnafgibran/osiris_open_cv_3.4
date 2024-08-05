cmake_minimum_required (VERSION 3.6)

project (Osiris)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

link_libraries(stdc++fs)

set(srcs 
    src/OsiCircle.cpp
    src/OsiEye.cpp
    src/OsiMain.cpp
    src/OsiManager.cpp
    src/OsiProcessings.cpp
    )
set(incs
    src/OsiCircle.h
    src/OsiEye.h
    src/OsiManager.h
    src/OsiProcessings.h
    src/OsiStringUtils.h
    )

# Add include directory
include_directories(include)

find_package(OpenCV QUIET)
find_package(ZLIB REQUIRED)

if (OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(${ZLIB_INCLUDE_DIRS})

  add_executable(Osiris ${srcs} ${incs})
  target_link_libraries(Osiris ${OpenCV_LIBS} ZLIB::ZLIB)
else()
  message("OpenCV not found, so we won't build the Osiris.")
endif()

if (MSVC)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/ALL_BUILD.vcxproj.user")
        file(GENERATE
            OUTPUT "${CMAKE_BINARY_DIR}/ALL_BUILD.vcxproj.user"
            INPUT "${CMAKE_SOURCE_DIR}/cmake/ALL_BUILD.vcxproj.user.in")
    endif()
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/Osiris.vcxproj.user")
        file(GENERATE
            OUTPUT "${CMAKE_BINARY_DIR}/Osiris.vcxproj.user"
            INPUT "${CMAKE_SOURCE_DIR}/cmake/Osiris.vcxproj.user.in")
    endif()
endif()
