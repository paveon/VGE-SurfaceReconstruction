#define SDL_MAIN_HANDLED

#include <iostream>
#include <tuple>

// #include <pcl/common/vector_average.h>
// #include <pcl/Vertices.h>
// #include <pcl/common/common_headers.h>
// #include <pcl/features/normal_3d.h>

// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/search/impl/search.hpp>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>
//#include <glm/gtx/string_cast.hpp>

#include "CloudModel.h"
#include "ExampleClouds.h"


static std::string g_ShaderFolder;

static const std::array<VertexRGB, 34> g_CoordVertices{
        VertexRGB(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(4.3f, 0.0f, 0.3f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(4.3f, 0.0f, -0.3f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(5.5f, 0.0f, 0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(6.0f, 0.0f, -0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(6.0f, 0.0f, 0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),
        VertexRGB(glm::vec3(5.5f, 0.0f, -0.25f), glm::vec3(0.0f, 0.0f, 1.0f)),

        VertexRGB(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 4.3f, 0.3f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 4.3f, -0.3f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 5.5f, -0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 6.0f, 0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 5.75f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 6.0f, -0.15f), glm::vec3(0.0f, 1.0f, 0.0f)),

        VertexRGB(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.3f, 0.0f, 4.3f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(-0.3f, 0.0f, 4.3f), glm::vec3(1.0f, 0.0f, 0.0f)),

        VertexRGB(glm::vec3(0.15f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(-0.15f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(-0.15f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.1f, 0.0f, 5.9f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(0.15f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
        VertexRGB(glm::vec3(-0.1f, 0.0f, 5.6f), glm::vec3(1.0f, 0.0f, 0.0f)),
};


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

enum Buffers {
    Ray,
    Coord,
    BUFFER_COUNT
};


int main(int /*argc*/, char ** /*argv*/) {
    int threadCount = omp_get_max_threads();
//    omp_set_num_threads
//    omp_get_thread_num
//    omp_get_num_threads
    std::cout << "[Thread count] " << threadCount << std::endl;

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

    std::array<CloudModel, 2> models{
            CloudModel(bunnyCloud(), g_ShaderFolder),
            CloudModel(sphereCloud(0.5f), g_ShaderFolder)
    };

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
        glNamedBufferData(VBOs[Coord], sizeof(VertexRGB) * g_CoordVertices.size(), g_CoordVertices.data(),
                          GL_STATIC_DRAW);
        glEnableVertexArrayAttrib(VAOs[Coord], 0);
        glEnableVertexArrayAttrib(VAOs[Coord], 1);
        glVertexArrayAttribFormat(VAOs[Coord], 0, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, pos));
        glVertexArrayAttribFormat(VAOs[Coord], 1, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, color));
        glVertexArrayVertexBuffer(VAOs[Coord], 0, VBOs[Coord], 0, sizeof(VertexRGB));
        glVertexArrayVertexBuffer(VAOs[Coord], 1, VBOs[Coord], 0, sizeof(VertexRGB));

        ImGui::GetIO().WantTextInput = true;
        ImGui::GetIO().WantCaptureKeyboard = true;
    });

    glm::vec3 color(0.0, 1.0, 0.0);
    glm::vec3 wireframeColor(1.0, 0.0, 0.0);

    app.addResizeCallback([&](int width, int height) {
        cam.setAspect((float) width / (float) height);
        glViewport(0, 0, width, height);
    });

    int neighbourhoodSize = 3;
    int currentModel = 0;
    int gridResX = 10;
    int gridResY = 10;
    int gridResZ = 10;
    std::array<const char *, 2> modelNames{
            "Bunny",
            "Sphere"
    };

    int methodID = (int) ReconstructionMethod::ModifiedHoppe;
    std::array<const char *, 4> methodLabels{
            "Modified Hoppe",
            "MLS",
            "PCL: Hoppe's",
            "PCL: Poisson"
    };

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

        if (rayCast) {
            glBindVertexArray(VAOs[Ray]);
            glDrawArrays(GL_LINES, 0, 2);
        }

        models[currentModel].Draw(pvm, color);

        // GUI
        label("FPS: " + std::to_string(ImGui::GetIO().Framerate));
        ImGui::Begin("Options", nullptr, ImVec2(300, 500));

        if (ImGui::Combo("Model", &currentModel, modelNames.data(), modelNames.size())) {
            gridResX = models[currentModel].m_Grid.GetResX();
            gridResY = models[currentModel].m_Grid.GetResY();
            gridResZ = models[currentModel].m_Grid.GetResZ();
        }

        ImGui::Combo("Method", &methodID, methodLabels.data(), methodLabels.size());

        auto method = static_cast<ReconstructionMethod>(methodID);
        if (ImGui::Button("Reconstruct"))
            models[currentModel].Reconstruct(method);

        switch (method) {
            case ReconstructionMethod::PCL_Hoppe:
                ImGui::SliderFloat("Ignore distance", &models[currentModel].m_IgnoreDistance, -1.0f, 1.0f);
                ImGui::SliderFloat("Iso level", &models[currentModel].m_IsoLevel, -0.1f, 0.1f);

            case ReconstructionMethod::ModifiedHoppe:
                ImGui::Text("Grid resolution");

                // Cannot merge due to short circuit || evaluation. Causes flickering
                // when moving the slider because the render command is skipped
                if (ImGui::SliderInt("X", &gridResX, 5, 100))
                    models[currentModel].m_Grid.SetResX(gridResX);
                if (ImGui::SliderInt("Y", &gridResY, 5, 100))
                    models[currentModel].m_Grid.SetResY(gridResY);
                if (ImGui::SliderInt("Z", &gridResZ, 5, 100))
                    models[currentModel].m_Grid.SetResZ(gridResZ);
                if (ImGui::Button("Regenerate grid"))
                    models[currentModel].m_Grid.Regenerate();

                if (method == ReconstructionMethod::ModifiedHoppe) {
                    ImGui::Text("Neighbourhood size");
                    if (ImGui::SliderInt("", &neighbourhoodSize, 1, 10)) {
                        models[currentModel].m_NeighbourhoodSize = (size_t)neighbourhoodSize;
                    }
                }
                break;
            case ReconstructionMethod::MLS:
                ImGui::Text("Grid resolution");

                // Cannot merge due to short circuit || evaluation. Causes flickering
                // when moving the slider because the render command is skipped
                if (ImGui::SliderInt("X", &gridResX, 5, 100))
                    models[currentModel].m_Grid.SetResX(gridResX);
                if (ImGui::SliderInt("Y", &gridResY, 5, 100))
                    models[currentModel].m_Grid.SetResY(gridResY);
                if (ImGui::SliderInt("Z", &gridResZ, 5, 100))
                    models[currentModel].m_Grid.SetResZ(gridResZ);
                if (ImGui::Button("Regenerate grid"))
                    models[currentModel].m_Grid.Regenerate();

                if (method == ReconstructionMethod::ModifiedHoppe || method == ReconstructionMethod::MLS) {
                    ImGui::Text("Neighbourhood size");
                    if (ImGui::SliderInt("", &neighbourhoodSize, 1, 25)) {
                        models[currentModel].m_NeighbourhoodSize = (size_t)neighbourhoodSize;
                    }
                }
                break;


            case ReconstructionMethod::PCL_Poisson:
                ImGui::SliderInt("Degree", &models[currentModel].m_Degree, 1, 5);
                ImGui::SliderInt("Depth", &models[currentModel].m_Depth, 1, 10);
                ImGui::SliderInt("Minimum depth", &models[currentModel].m_MinDepth, 1, 10);
                ImGui::SliderInt("Iso divide", &models[currentModel].m_IsoDivide, 1, 16);
                ImGui::SliderInt("Solver divide", &models[currentModel].m_SolverDivide, 1, 16);
                ImGui::SliderFloat("Point weight", &models[currentModel].m_PointWeight, 1.0f, 10.0f);
                ImGui::SliderFloat("Samples per node", &models[currentModel].m_SamplesPerNode, 1.0f, 10.0f);
                ImGui::SliderFloat("Scale", &models[currentModel].m_Scale, 1.1f, 5.0f);
                break;
        }

        ImGui::Text("Debug Options");
        ImGui::Checkbox("Show mesh", &models[currentModel].m_ShowMesh);
        ImGui::Checkbox("Show grid", &models[currentModel].m_ShowGrid);
        ImGui::Checkbox("Show input PC", &models[currentModel].m_ShowInputPC);
        if (models[currentModel].m_ShowInputPC) {
            ImGui::Checkbox("Show normals", &models[currentModel].m_ShowNormals);
        }

        ImGui::End();

        if (showErrorPopup) {
            if (ImGui::BeginPopupContextVoid("Error")) {
                ImGui::Text("%s", errorBuffer);
                ImGui::EndPopup();
            }
        }

        if (showDemo) {
            ImGui::ShowTestWindow(&showDemo);
        }
    });

    // TODO: implement some shortcuts in the future
    app.addKeyPressCallback([&](SDL_Keycode code, uint16_t) {
        switch (code) {
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
        if (button == 3) {
            std::array<glm::vec3, 2> rayVertices{rayOrigin, rayOrigin + (rayDir * 100.0f)};
            glNamedBufferSubData(VBOs[Ray], 0, sizeof(glm::vec3) * 2, rayVertices.data());
            rayCast = true;
        }
    });

    app.addMouseMoveCallback([&](int, int, int x, int y) {
        auto[origin, dir] = castRay(x, y);
        rayOrigin = origin;
        rayDir = dir;
    });

    return app.run();
}
