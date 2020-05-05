#define SDL_MAIN_HANDLED

#include <iostream>
#include <tuple>

// #include <pcl/common/vector_average.h>
// #include <pcl/Vertices.h>
// #include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/conversions.h>

// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/search/impl/search.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>
//#include <glm/gtx/string_cast.hpp>

#include "CloudModel.h"
#include "ExampleClouds.h"


static std::string g_ShaderFolder;
static std::string g_ModelFolder;

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


enum Buffers {
    Ray,
    Coord,
    BUFFER_COUNT
};

pcl::PointCloud<pcl::PointNormal>::Ptr normalizeCloud(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
{
    float increase = 0.1f;
    Eigen::Vector4f min, max;
    pcl::getMinMax3D(*cloud, min, max);
    Eigen::Vector4f size = max - min;
    min -= (size * (increase / 2.0f));
    max += (size * (increase / 2.0f));
    size *= (1.0f + increase);

    float scaleFactor = 1.0f / size.x();
    pcl::PointCloud<pcl::PointNormal>::Ptr normalized(new pcl::PointCloud<pcl::PointNormal>());
    Eigen::Affine3f transform(Eigen::Translation3f(-min.x(), -min.y(), -min.z()));
    Eigen::Matrix4f matrix = transform.matrix() * scaleFactor;
    pcl::transformPointCloud(*cloud, *normalized, matrix);

    return normalized;
}


pcl::PointCloud<pcl::PointNormal>::Ptr loadModel(std::string modelPath) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PLYReader reader;
    std::string filepath(g_ModelFolder + modelPath);

    std::cout << "Loading ply file: " << filepath << std::endl;
    if (pcl::io::loadPLYFile(filepath, *cloud) != -1) {
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>());

        normalEstimation.setInputCloud(cloud);
        normalEstimation.setSearchMethod(tree);
        normalEstimation.setKSearch(5);
        // normalEstimation.setRadiusSearch(0.03); // Use all neighbors in a sphere of radius 3cm
        normalEstimation.compute(*cloudNormals); // Compute the features

        pcl::PointCloud<pcl::PointNormal>::Ptr result(new pcl::PointCloud<pcl::PointNormal>());
        pcl::concatenateFields(*cloud, *cloudNormals, *result);

        return result;
    }
    throw std::runtime_error("Model could not be loaded");
}


int main(int /*argc*/, char ** /*argv*/) {
    int threadCount = omp_get_max_threads();
    std::cout << "[Thread count] " << threadCount << std::endl;

    BaseApp app;
    PerspectiveCamera cam;
    OrbitManipulator manipulator(&cam);
    manipulator.setZoom(5.0f);
    manipulator.setRotationX(-90.0f);
    manipulator.setupCallbacks(app);

    g_ShaderFolder = app.getResourceDir() + "Shaders/";
    g_ModelFolder = app.getResourceDir() + "Models/";

    std::array<CloudModel, 15> models{
            CloudModel("Bunny", normalizeCloud(bunnyCloud()), g_ShaderFolder),
            CloudModel("Sphere", normalizeCloud(sphereCloud(0.5f)), g_ShaderFolder),

            CloudModel("Bunny 3.0 MB", normalizeCloud(loadModel("bunny/bun_zipper.ply")), g_ShaderFolder),
            CloudModel("Bunny 0.7 MB", normalizeCloud(loadModel("bunny/bun_zipper_res2.ply")), g_ShaderFolder),
            CloudModel("Bunny 0.2 MB", normalizeCloud(loadModel("bunny/bun_zipper_res3.ply")), g_ShaderFolder),
            CloudModel("Bunny 0.03 MB", normalizeCloud(loadModel("bunny/bun_zipper_res4.ply")), g_ShaderFolder),

            CloudModel("Drill VRIP", normalizeCloud(loadModel("drill/drill_shaft_vrip.ply")), g_ShaderFolder),
            CloudModel("Drill Zipper", normalizeCloud(loadModel("drill/drill_shaft_zip.ply")), g_ShaderFolder),

            CloudModel("Buddha 10.9 MB", normalizeCloud(loadModel("happy_recon/happy_vrip_res2.ply")), g_ShaderFolder),
            CloudModel("Buddha 2.4 MB", normalizeCloud(loadModel("happy_recon/happy_vrip_res3.ply")), g_ShaderFolder),
            CloudModel("Buddha 0.5 MB", normalizeCloud(loadModel("happy_recon/happy_vrip_res4.ply")), g_ShaderFolder),

            CloudModel("Dragon 7.3 MB", normalizeCloud(loadModel("dragon_recon/dragon_vrip_res2.ply")), g_ShaderFolder),
            CloudModel("Dragon 1.7 MB", normalizeCloud(loadModel("dragon_recon/dragon_vrip_res3.ply")), g_ShaderFolder),
            CloudModel("Dragon 0.4 MB", normalizeCloud(loadModel("dragon_recon/dragon_vrip_res4.ply")), g_ShaderFolder),

            CloudModel("Armadillo", normalizeCloud(loadModel("Armadillo.ply")), g_ShaderFolder),
    };

    std::array<const char *, models.size()> modelNames{};
    for (size_t i = 0; i < models.size(); ++i) {
        modelNames[i] = models[i].Name().data();
    }

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

    double reconstructionTime = 0.0f;
    int neighbourhoodSize = 3;
    int currentModel = 0;
    int gridResX = 10;
    int gridResY = 10;
    int gridResZ = 10;

    int methodID = (int) ReconstructionMethod::ModifiedHoppe;

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

        ImGui::Text("Point count: %lu", models[currentModel].CloudSize());
        ImGui::Text("Triangle count: %lu", models[currentModel].TriangleCount());
        ImGui::Text("Reconstruction time: %f", reconstructionTime);

        ImGui::Combo("Method", &methodID, s_MethodLabels.data(), s_MethodLabels.size());

        auto method = static_cast<ReconstructionMethod>(methodID);
        if (ImGui::Button("Reconstruct"))
            reconstructionTime = models[currentModel].Reconstruct(method);

        auto renderGridGUI = [&]() {
            // Cannot merge due to short circuit || evaluation. Causes flickering
            // when moving the slider because the render command is skipped
            ImGui::Text("Grid resolution");
            if (ImGui::SliderInt("X", &gridResX, 5, 100))
                models[currentModel].m_Grid.SetResX(gridResX);
            if (ImGui::SliderInt("Y", &gridResY, 5, 100))
                models[currentModel].m_Grid.SetResY(gridResY);
            if (ImGui::SliderInt("Z", &gridResZ, 5, 100))
                models[currentModel].m_Grid.SetResZ(gridResZ);
            if (ImGui::Button("Regenerate grid"))
                models[currentModel].m_Grid.Regenerate();
        };

        switch (method) {
            case ReconstructionMethod::PCL_Hoppe:
                ImGui::SliderFloat("Iso level", &models[currentModel].m_IsoLevel, 0.0f, 1.0f);
                renderGridGUI();
                break;

            case ReconstructionMethod::PCL_MarchingCubesRBF:
                ImGui::Text("Off surface displacement");
                ImGui::SliderFloat("", &models[currentModel].m_OffSurfaceDisplacement, 0.0f, 1.0f);
                ImGui::SliderFloat("Iso level", &models[currentModel].m_IsoLevel, 0.0f, 1.0f);
                renderGridGUI();
                break;

            case ReconstructionMethod::ModifiedHoppe:
                ImGui::Text("Neighbourhood size");
                if (ImGui::SliderInt("", &neighbourhoodSize, 1, 10))
                    models[currentModel].m_NeighbourhoodSize = (size_t)neighbourhoodSize;
                
                renderGridGUI();
                break;

            case ReconstructionMethod::PCL_Poisson:
                ImGui::SliderInt("Depth", &models[currentModel].m_Depth, 1, 10);
                ImGui::SliderInt("Minimum depth", &models[currentModel].m_MinDepth, 1, 10);
                ImGui::SliderInt("Iso divide", &models[currentModel].m_IsoDivide, 1, 16);
                ImGui::SliderInt("Solver divide", &models[currentModel].m_SolverDivide, 1, 16);
                ImGui::SliderFloat("Point weight", &models[currentModel].m_PointWeight, 1.0f, 10.0f);
                ImGui::SliderFloat("Samples per node", &models[currentModel].m_SamplesPerNode, 1.0f, 10.0f);
                ImGui::SliderFloat("Scale", &models[currentModel].m_Scale, 1.1f, 5.0f);
                break;

            case ReconstructionMethod::PCL_ConcaveHull:
                ImGui::SliderFloat("Alpha", &models[currentModel].m_Alpha, 0.00001f, 0.2f);
                break;

            case ReconstructionMethod::PCL_ConvexHull: 
                break;

            case ReconstructionMethod::PCL_GreedyProjectionTriangulation:
                ImGui::SliderInt("Maximum NN", &models[currentModel].m_MaxNN, 1, 300);
                ImGui::SliderFloat("Max surface angle", &models[currentModel].m_MaxSurfaceAngle, 1.0f, 150.0f);
                ImGui::SliderFloat("Max triangle angle", &models[currentModel].m_MaxAngle, 1.0f, 150.0f);
                ImGui::SliderFloat("Min triangle angle", &models[currentModel].m_MinAngle, 1.0f, 120.0f);
                ImGui::SliderFloat("Search radius", &models[currentModel].m_SearchRadius, 0.1f, 10.0f);
                ImGui::SliderFloat("Mu", &models[currentModel].m_Mu, 0.1f, 10.0f);
                break;

            case ReconstructionMethod::PCL_OrganizedFastMesh:
                ImGui::SliderFloat("Angle tolerance", &models[currentModel].m_AngleTolerance, 1.0f, 90.0f);
                ImGui::SliderFloat("Distance tolerance", &models[currentModel].m_DistTolerance, 0.0f, 10.0f);
                ImGui::SliderFloat("Max edge length", &models[currentModel].m_A, 0.001f, 1.0f);
                break;
        }

        ImGui::Text("Debug Options");
        ImGui::Checkbox("Show mesh", &models[currentModel].m_ShowMesh);
        ImGui::Checkbox("Show grid", &models[currentModel].m_ShowGrid);
        ImGui::Checkbox("Show BB", &models[currentModel].m_ShowBB);
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
