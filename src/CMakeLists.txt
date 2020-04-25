project(SurfaceReconstruction)

file(GLOB shaders ${VGE_SOURCE_DIR}/Resources/Shaders/*)
file(GLOB src ${VGE_SOURCE_DIR}/src/*)
source_group("Shaders" FILES ${shaders})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
# include_directories(
#         ../extern/pcl-1.7
#         /usr/include/eigen3
#         /usr/include/ni
# #        /usr/include/vtk-6.2
#         /usr/include/freetype2
#         /usr/include/x86_64-linux-gnu/freetype2
#         /usr/include/jsoncpp
#         /usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent
#         /usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include
#         /usr/lib/openmpi/include
#         /usr/lib/openmpi/include/openmpi
# #        /usr/include/python2.7
#         /usr/include/x86_64-linux-gnu
#         /usr/include/hdf5/serial
#         /usr/include/libxml2
#         /usr/include/tcl
#         )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(${PROJECT_NAME} ${src} ${shaders})
target_link_libraries(${PROJECT_NAME} PUBLIC Framework imgui ${PCL_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)