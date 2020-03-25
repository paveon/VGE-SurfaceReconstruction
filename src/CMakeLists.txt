project(SurfaceReconstruction)

file(GLOB shaders ${VGE_SOURCE_DIR}/Resources/Shaders/*)
file(GLOB src ${VGE_SOURCE_DIR}/src/*)
source_group("Shaders" FILES ${shaders})

add_executable(${PROJECT_NAME} ${src} ${shaders})
target_link_libraries(${PROJECT_NAME} PUBLIC Framework imgui)
#target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
