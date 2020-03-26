project(SurfaceReconstruction)

file(GLOB shaders ${VGE_SOURCE_DIR}/Resources/Shaders/*)
file(GLOB src ${VGE_SOURCE_DIR}/src/*)
source_group("Shaders" FILES ${shaders})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(${PROJECT_NAME} ${src} ${shaders})
target_link_libraries(${PROJECT_NAME} PUBLIC Framework imgui ${PCL_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)