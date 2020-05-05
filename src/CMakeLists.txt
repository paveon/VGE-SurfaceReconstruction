project(SurfaceReconstruction)

file(GLOB shaders ${VGE_SOURCE_DIR}/Resources/Shaders/*)
file(GLOB src ${VGE_SOURCE_DIR}/src/*)
source_group("Shaders" FILES ${shaders})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

find_package(OpenMP)

add_executable(${PROJECT_NAME} ${src} ${shaders})
target_link_libraries(${PROJECT_NAME} PUBLIC Framework imgui ${PCL_LIBRARIES})

IF (OpenMP_FOUND)
    IF (CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF ()
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
ENDIF ()