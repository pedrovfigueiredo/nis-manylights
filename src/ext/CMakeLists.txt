
cmake_minimum_required (VERSION 3.12)

if (MSVC)
  # add_definitions(/D _CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup
  #   /wd4018 /wd4100 /wd4101 /wd4127 /wd4146 /wd4232 /wd4242 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309 
  #   /wd4310 /wd4334 /wd4456 /wd4464 /wd4668 /wd4701 /wd4703 /wd4711 /wd4756 /wd4820 /wd5045 /wd5250)
  add_definitions(/D _CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup)
endif ()

###########################################################################
# tcnn
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(tcnn)
set(TCNN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/tcnn/include PARENT_SCOPE)
set(TCNN_DEPENDENCIES 
  ${CMAKE_CURRENT_SOURCE_DIR}/tcnn/dependencies 
  ${CMAKE_CURRENT_SOURCE_DIR}/tcnn/dependencies/fmt/include 
  PARENT_SCOPE)
set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
# include_directories(${TCNN_INCLUDE})
# include_directories(${TCNN_DEPENDENCIES})

###########################################################################
# stb

set (STB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)

###########################################################################
# filesystem

set (FILESYSTEM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/filesystem PARENT_SCOPE)

###########################################################################
# libdeflate

add_subdirectory (libdeflate)

set (LIBDEFLATE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libdeflate PARENT_SCOPE)
set (LIBDEFLATE_LIBRARIES deflate::deflate PARENT_SCOPE)

set_property (TARGET deflate PROPERTY FOLDER "ext")

###########################################################################
# zlib

find_package (ZLIB)
if (NOT ZLIB_FOUND)
  # Build zlib
  set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory (zlib)

  set (ZLIB_LIBRARIES zlibstatic)
  set (ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

  # try to make openexr happy about this...
  set (ZLIB_LIBRARY zlibstatic)
  set (ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)
  set (ZLIB_FOUND TRUE)

  set_property (TARGET zlibstatic PROPERTY FOLDER "ext")

  add_library (ZLIB::ZLIB ALIAS zlibstatic)
  include_directories(${ZLIB_INCLUDE_DIRS})  # yuck, but so openexr/ptex can find zlib.h...
endif ()

set (ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)
set (ZLIB_LIBRARIES ${ZLIB_LIBARIES} PARENT_SCOPE)

###########################################################################
# OpenEXR

# find_package (OpenEXR)

if (OPENEXR_FOUND)
  message (STATUS "Found OpenEXR version ${OPENEXR_VERSION}")
  set (OPENEXR_INCLUDE ${OPENEXR_INCLUDES}/OpenEXR ${ILMBASE_INCLUDES}/OpenEXR PARENT_SCOPE)
  if (WIN32)
    set (OPENEXR_LIBS ${OPENEXR_LIBRARIES} ${ZLIB_LIBRARY} PARENT_SCOPE)
  else ()
    set (OPENEXR_LIBS ${OPENEXR_LIBRARIES} PARENT_SCOPE)
  endif ()
else ()
  message (STATUS "OpenEXR not found; building it from scratch.")

  SET (CMAKE_POLICY_DEFAULT_CMP0077 NEW)

  set (ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
  set (OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
  set (OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
  set (ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
  set (PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
  set (OPENEXR_BUILD_UTILS          OFF CACHE BOOL " " FORCE)

  add_subdirectory (openexr)

  set_property (TARGET IexMath IlmThread Half
                Iex Imath IlmImf HalfTest IexTest
                IlmImfExamples IlmImfTest IlmImfUtil IlmImfUtilTest ImathTest
                PROPERTY FOLDER "ext/OpenEXR")

  set (OPENEXR_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Imath
    ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Half
    ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Iex
    ${CMAKE_CURRENT_SOURCE_DIR}/openexr/OpenEXR/IlmImf
    ${CMAKE_CURRENT_BINARY_DIR}/openexr/IlmBase/config
    ${CMAKE_CURRENT_BINARY_DIR}/openexr/OpenEXR/config
  PARENT_SCOPE
  )

  if (WIN32)
    set (OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half ${ZLIB_LIBRARY} PARENT_SCOPE)
  else ()
    set (OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half PARENT_SCOPE)
  endif ()
endif ()

###########################################################################
# ptex

set (PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

set (CMAKE_MACOSX_RPATH 1)
if (WIN32)
  add_definitions (/DPTEX_STATIC)
endif ()

add_subdirectory (ptex)

set_property (TARGET Ptex_static ptxinfo halftest ftest rtest wtest PROPERTY FOLDER "ext/ptex")

set (PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)

###########################################################################
# double-conversion

add_subdirectory (double-conversion)

set (DOUBLE_CONVERSION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion PARENT_SCOPE)

set_property (TARGET double-conversion cctest PROPERTY FOLDER "ext")

###########################################################################
# nanovdb

set (NANOVDB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb PARENT_SCOPE)

###########################################################################
# FLIP

set (FLIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/flip PARENT_SCOPE)

add_library (flip_lib STATIC ${CMAKE_CURRENT_SOURCE_DIR}/flip/flip.cpp)

set_property (TARGET flip_lib PROPERTY FOLDER "ext")

###########################################################################
# utf8proc

set (UTF8PROC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/utf8proc PARENT_SCOPE)

add_subdirectory (utf8proc)

###########################################################################
# qoi

set (QOI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/qoi PARENT_SCOPE)

###########################################################################
# glfw / glad

set (GLFW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include PARENT_SCOPE)
set (GLAD_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glad/include PARENT_SCOPE)

set (GLFW_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set (GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory (glfw)
add_subdirectory (glad)

set_property (TARGET glfw PROPERTY FOLDER "ext")
set_property (TARGET glad PROPERTY FOLDER "ext")
