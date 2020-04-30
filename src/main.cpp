#define SDL_MAIN_HANDLED

#include <iostream>
#include <thread>
#include <chrono>
#include <tuple>

#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>

//#include <pcl/surface/3rdparty/poisson4/octree_poisson.h>
//#include <pcl/surface/3rdparty/poisson4/sparse_matrix.h>
//#include <pcl/surface/3rdparty/poisson4/function_data.h>
//#include <pcl/surface/3rdparty/poisson4/ppolynomial.h>
//#include <pcl/surface/3rdparty/poisson4/multi_grid_octree_data.h>
//#include <pcl/surface/3rdparty/poisson4/geometry.h>

#include <pcl/point_types.h>
// #include <pcl/common/common_headers.h>
// #include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>

// #include <pcl/surface/poisson.h>
// #include <pcl/surface/marching_cubes_hoppe.h>

#include <glm/gtx/string_cast.hpp>
#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>
#include "bunny.h"

#include "CloudModel.h"


static std::string g_ShaderFolder;

static const std::array<BasicVertex, 34> g_CoordVertices{
    BasicVertex(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(4.3f, 0.0f, 0.3f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(4.3f, 0.0f, -0.3f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(5.5f, 0.0f, 0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(6.0f, 0.0f, -0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(6.0f, 0.0f, 0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
    BasicVertex(glm::vec3(5.5f, 0.0f, -0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),

    BasicVertex(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 4.3f, 0.3f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 4.3f, -0.3f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 5.5f, -0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 6.0f, 0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 5.75f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 6.0f, -0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),

    BasicVertex(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.3f, 0.0f, 4.3f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(-0.3f, 0.0f, 4.3f), glm::vec3(1.0f, 0.0f, 0.0f)),

    BasicVertex(glm::vec3(0.15f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(-0.15f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(-0.15f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.1f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(0.15f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
    BasicVertex(glm::vec3(-0.1f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
};


// pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithOutNormals()
// {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point(new pcl::PointCloud<pcl::PointXYZ>());
//     for (const auto &bunnyVertice : bunnyVertices)
//     {
//         auto *pt = new pcl::PointXYZ;
//         pt->x = bunnyVertice.position[0];
//         pt->y = bunnyVertice.position[1];
//         pt->z = bunnyVertice.position[2];
//         cloud_point->push_back(*pt);
//     }
//     // Create the normal estimation class, and pass the input dataset to it
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
//     normalEstimation.setInputCloud(cloud_point);
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//     normalEstimation.setSearchMethod(tree);
//     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//     // Use all neighbors in a sphere of radius 3cm
//     normalEstimation.setRadiusSearch(0.03);
//     // Compute the features
//     normalEstimation.compute(*cloud_normals);
//     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals(new pcl::PointCloud<pcl::PointNormal>());
//     pcl::concatenateFields(*cloud_point, *cloud_normals, *cloud_point_normals);
//     std::cout << cloud_point_normals->size() << std::endl;
//     return cloud_point_normals;
// }

// void show(const pcl::PolygonMesh &mesh)
// {
//     // --------------------------------------------
//     // -----Open 3D viewer and add point cloud-----
//     // --------------------------------------------
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//     viewer->setBackgroundColor(0, 0, 0);
//     viewer->setCameraPosition(0, 0, 0, 0, 0, 0);
//     viewer->addPolygonMesh(mesh);
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//     viewer->addCoordinateSystem(1.0);
//     viewer->initCameraParameters();
//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce(100);
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }
// }

// pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
// {
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> *ne = new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>;
//     ne->setInputCloud(cloud);

//     // Create an empty kdtree representation, and pass it to the normal estimation object.
//     // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//     ne->setSearchMethod(tree);

//     // Output datasets
//     pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

//     // Use all neighbors in a sphere of radius 3cm
//     ne->setRadiusSearch(0.3);

//     // Compute the features
//     ne->compute(*cloud_normals);
//     return cloud_normals;
// }

enum Buffers
{
    Ray,
    Coord,
    BUFFER_COUNT
};


int main(int /*argc*/, char ** /*argv*/)
{
    // Simple caching system for later (we might want to display multiple models during presentation?)
    // std::ifstream indexCache("index_cache.dat", std::ios::in | std::ios::binary);
    // if (false && pcl::io::loadPCDFile<pcl::PointNormal>("surface_cache.pcd", *surfaceCloud) >= 0 && indexCache.is_open())
    // {
    //     // Cache exists
    //     size_t cacheSize;
    //     indexCache.read((char *)&cacheSize, sizeof(size_t));
    //     flatIndices.resize(cacheSize);
    //     indexCache.read((char *)flatIndices.data(), cacheSize * sizeof(GLuint));
    // }
    // else
    // {
    //     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals;
    //     // if (false)
    //     //     cloud_point_normals = bunnyWithOutNormals();
    //     // else
    //     //     cloud_point_normals = bunnyWithNormals();

    //     // pcl::Poisson<pcl::PointNormal> poisson;
    //     // poisson.setDepth(6); // Was painfully slow with depth of 12
    //     // poisson.setInputCloud(cloud_point_normals);

    //     std::vector<pcl::Vertices> outputIndices;

    //     // Reconstruction
    //     pcl::MarchingCubesHoppe<pcl::PointNormal> hoppe;
    //     hoppe.setInputCloud(cloud_point_normals);
    //     hoppe.setGridResolution(40, 40, 40);
    //     hoppe.reconstruct(*surfaceCloud, outputIndices);
    //     poisson.reconstruct(*surfaceCloud, outputIndices);

    //     // PCL uses weird memory layout for indices, copy the data
    //     // into new vector with flat layout so that it can be used by OpenGL
    //     size_t idx = 0;
    //     flatIndices.resize(outputIndices.size() * 3);
    //     for (size_t i = 0; i < outputIndices.size(); i++)
    //     {
    //         const auto &indices = outputIndices[i];
    //         for (size_t index : indices.vertices)
    //             flatIndices[idx++] = index;
    //     }

    //     // Save cache
    //     pcl::io::savePCDFileASCII("surface_cache.pcd", *surfaceCloud);
    //     std::ofstream cacheFile("index_cache.dat", std::ios::out | std::ios::binary);
    //     size_t cacheSize = flatIndices.size();
    //     cacheFile.write((char *)&cacheSize, sizeof(size_t));
    //     cacheFile.write((char *)flatIndices.data(), cacheSize * sizeof(GLuint));
    //     cacheFile.close();
    // }
    // show(mesh);

    BaseApp app;
    PerspectiveCamera cam;
    OrbitManipulator manipulator(&cam);
    manipulator.setZoom(5.0f);
    manipulator.setRotationX(-90.0f);
    manipulator.setupCallbacks(app);

    g_ShaderFolder = app.getResourceDir() + "Shaders/";


    CloudModel bunnyModel(g_ShaderFolder);

    // TODO: Move to CloudModel later on
    // std::vector<GLuint> flatIndices;
    // pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    // auto &surfacePoints = surfaceCloud->points;
    // std::cout << "Cloud size: " << surfacePoints.size() << std::endl;
    // std::cout << "Triangle count: " << (flatIndices.size() / 3) << std::endl;

    std::array<GLuint, BUFFER_COUNT> VAOs{};
    std::array<GLuint, BUFFER_COUNT> VBOs{};
    std::array<GLuint, BUFFER_COUNT> EBOs{};
    ProgramObject wireframeProgram;

    auto mainWindow = app.getMainWindow();

    bool showErrorPopup = false;
    bool showDemo = false;
    bool rayCast = false;

    ImVec2 buttonSize(120, 20);

    char errorBuffer[256];

    app.addInitCallback([&]() {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(0.2, 0.2, 0.2, 1);
        // glEnable(GL_ALPHA_TEST);
        // glAlphaFunc(GL_GREATER,0.0f);

        auto wireframeVS = compileShader(GL_VERTEX_SHADER, Loader::text(g_ShaderFolder + "wireframe.vert"));
        auto wireframeFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(g_ShaderFolder + "wireframe.frag"));
        wireframeProgram = createProgram(wireframeVS, wireframeFS);

        // Create buffers
        glCreateVertexArrays(VAOs.size(), VAOs.data());
        glCreateBuffers(VBOs.size(), VBOs.data());
        glCreateBuffers(EBOs.size(), EBOs.data());

        glNamedBufferData(VBOs[Ray], sizeof(glm::vec3) * 2, 0, GL_DYNAMIC_DRAW);
        glEnableVertexArrayAttrib(VAOs[Ray], 0);
        glVertexArrayAttribFormat(VAOs[Ray], 0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(VAOs[Ray], 0, VBOs[Ray], 0, sizeof(glm::vec3));

        // Buffers for coordinate system arrows
        glNamedBufferData(VBOs[Coord], sizeof(BasicVertex) * g_CoordVertices.size(), g_CoordVertices.data(), GL_STATIC_DRAW);
        glEnableVertexArrayAttrib(VAOs[Coord], 0);
        glEnableVertexArrayAttrib(VAOs[Coord], 1);
        glVertexArrayAttribFormat(VAOs[Coord], 0, 3, GL_FLOAT, GL_FALSE, offsetof(BasicVertex, pos));
        glVertexArrayAttribFormat(VAOs[Coord], 1, 3, GL_FLOAT, GL_FALSE, offsetof(BasicVertex, color));
        glVertexArrayVertexBuffer(VAOs[Coord], 0, VBOs[Coord], 0, sizeof(BasicVertex));
        glVertexArrayVertexBuffer(VAOs[Coord], 1, VBOs[Coord], 0, sizeof(BasicVertex));

        ImGui::GetIO().WantTextInput = true;
        ImGui::GetIO().WantCaptureKeyboard = true;
    });

    glm::vec3 color(0.0, 1.0, 0.0);
    glm::vec3 wireframeColor(1.0, 0.0, 0.0);

    app.addResizeCallback([&](int width, int height) {
        cam.setAspect((float)width / (float)height);
        glViewport(0, 0, width, height);
    });

    app.addDrawCallback([&]() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw coordinate arrows
        glm::mat4 pvm = cam.getProjection() * cam.getView();
        wireframeProgram.use();
        wireframeProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(VAOs[Coord]);
        glLineWidth(3.0f);
        glDrawArrays(GL_LINES, 0, g_CoordVertices.size());
        glLineWidth(1.0f);

        if (rayCast)
        {
            glBindVertexArray(VAOs[Ray]);
            glDrawArrays(GL_LINES, 0, 2);
        }

        bunnyModel.Draw(pvm, color);

        // GUI
        label("FPS: " + std::to_string(ImGui::GetIO().Framerate));
        ImGui::Begin("Options", nullptr, ImVec2(300, 500));
        ImGui::Checkbox("Show mesh", &bunnyModel.m_ShowMesh);
        ImGui::Text("Grid resolution");

        // Cannot merge due to short curcuit || evaluation. Causes flickering
        // when moving the slider because the render command is skiped
        if (ImGui::SliderInt("X", &bunnyModel.m_GridSizeX, 5, 100))
        {
            bunnyModel.m_InvalidatedGrid = true;
        }

        if (ImGui::SliderInt("Y", &bunnyModel.m_GridSizeY, 5, 100))
        {
            bunnyModel.m_InvalidatedGrid = true;
        }

        if (ImGui::SliderInt("Z", &bunnyModel.m_GridSizeZ, 5, 100))
        {
            bunnyModel.m_InvalidatedGrid = true;
        }

        if (ImGui::Button("Regenerate grid"))
        {
            bunnyModel.RegenerateGrid();
        }

        if (ImGui::Button("Reconstruct"))
        {
            bunnyModel.Reconstruct();
        }

        ImGui::Text("Debug Options");
        ImGui::Checkbox("Show grid", &bunnyModel.m_ShowGrid);
        ImGui::Checkbox("Show input PC", &bunnyModel.m_ShowInputPC);
        if (bunnyModel.m_ShowInputPC)
        {
            ImGui::Checkbox("Show normals", &bunnyModel.m_ShowNormals);
        }

        ImGui::Checkbox("Show connections", &bunnyModel.m_ShowConnections);
        ImGui::SliderInt("Connection index", &bunnyModel.m_ConnectionIdx, 0, bunnyModel.GetCornerCount() - 1);

        ImGui::End();

        if (showErrorPopup)
        {
            if (ImGui::BeginPopupContextVoid("Error"))
            {
                ImGui::Text("%s", errorBuffer);
                ImGui::EndPopup();
            }
        }

        if (showDemo)
        {
            ImGui::ShowTestWindow(&showDemo);
        }
    });

    // TODO: implement some shortcuts in the future
    app.addKeyPressCallback([&](SDL_Keycode code, uint16_t) {
        switch (code)
        {
        case SDLK_ESCAPE:
            break;
        case SDLK_LEFT:
            break;
        case SDLK_RIGHT:
            break;
        case SDLK_UP:
            break;
        case SDLK_DOWN:
            break;
        case SDLK_PAGEUP:
            break;
        case SDLK_PAGEDOWN:
            break;
        case SDLK_DELETE:
            break;
        }
    });

    // Raycasting, maybe will be useful later on?
    glm::vec3 rayOrigin;
    glm::vec3 rayDir;
    auto castRay = [&mainWindow, &cam](int x, int y) {
        float width = mainWindow->getWidth();
        float height = mainWindow->getHeight();
        float ndsX = (2.0f * x) / width - 1.0f;
        float ndsY = 1.0f - (2.0f * y) / height;
        glm::vec4 clipOrigin = glm::vec4(ndsX, ndsY, -1.0, 1.0f);
        glm::vec4 eyeOrigin = glm::inverse(cam.getProjection()) * clipOrigin;
        eyeOrigin.z = -1.0f;
        eyeOrigin.w = 0.0f;
        glm::vec3 rayDir(glm::normalize(glm::vec3(glm::inverse(cam.getView()) * eyeOrigin)));
        glm::vec3 eye(cam.getEye());
        return std::make_pair(eye, rayDir);
    };

    app.addMousePressCallback([&](uint8_t button, int, int) {
        if (button == 3)
        {
            std::array<glm::vec3, 2> rayVertices{rayOrigin, rayOrigin + (rayDir * 100.0f)};
            glNamedBufferSubData(VBOs[Ray], 0, sizeof(glm::vec3) * 2, rayVertices.data());
            rayCast = true;
        }
    });

    app.addMouseMoveCallback([&](int, int, int x, int y) {
        auto [origin, dir] = castRay(x, y);
        rayOrigin = origin;
        rayDir = dir;
    });

    return app.run();
}
