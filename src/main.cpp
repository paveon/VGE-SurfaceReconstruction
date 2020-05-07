#define SDL_MAIN_HANDLED

#include <iostream>
#include <tuple>

#include <pcl/features/normal_3d.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>

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
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PLYReader reader;
    std::string filepath(g_ModelFolder + modelPath);

    std::cout << "Loading ply file: " << filepath << std::endl;
    if (pcl::io::loadPLYFile(filepath, *cloud) != -1) {
        return normalizeCloud(cloud);
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

            CloudModel("Bunny 3.0 MB", loadModel("bunny/bun_zipper.ply"), g_ShaderFolder),
            CloudModel("Bunny 0.7 MB", loadModel("bunny/bun_zipper_res2.ply"), g_ShaderFolder),
            CloudModel("Bunny 0.2 MB", loadModel("bunny/bun_zipper_res3.ply"), g_ShaderFolder),
            CloudModel("Bunny 0.03 MB", loadModel("bunny/bun_zipper_res4.ply"), g_ShaderFolder),

            CloudModel("Drill VRIP", loadModel("drill/drill_shaft_vrip.ply"), g_ShaderFolder),
            CloudModel("Drill Zipper", loadModel("drill/drill_shaft_zip.ply"), g_ShaderFolder),

            CloudModel("Buddha 10.9 MB", loadModel("happy_recon/happy_vrip_res2.ply"), g_ShaderFolder),
            CloudModel("Buddha 2.4 MB", loadModel("happy_recon/happy_vrip_res3.ply"), g_ShaderFolder),
            CloudModel("Buddha 0.5 MB", loadModel("happy_recon/happy_vrip_res4.ply"), g_ShaderFolder),

            CloudModel("Dragon 7.3 MB", loadModel("dragon_recon/dragon_vrip_res2.ply"), g_ShaderFolder),
            CloudModel("Dragon 1.7 MB", loadModel("dragon_recon/dragon_vrip_res3.ply"), g_ShaderFolder),
            CloudModel("Dragon 0.4 MB", loadModel("dragon_recon/dragon_vrip_res4.ply"), g_ShaderFolder),

            CloudModel("Armadillo", loadModel("Armadillo.ply"), g_ShaderFolder),
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
    int currentModelIdx = 0;
    int gridResX = 10;
    int gridResY = 10;
    int gridResZ = 10;
    CloudModel* currentModel = &models[currentModelIdx];

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

        currentModel->Draw(pvm, color);

        // GUI
        label("FPS: " + std::to_string(ImGui::GetIO().Framerate));
        ImGui::Begin("Options", nullptr, ImVec2(300, 500));

        if (ImGui::Combo("Model", &currentModelIdx, modelNames.data(), modelNames.size())) {
            currentModel = &models[currentModelIdx];
            gridResX = currentModel->m_Grid.GetResX();
            gridResY = currentModel->m_Grid.GetResY();
            gridResZ = currentModel->m_Grid.GetResZ();
        }

        ImGui::Text("Point count: %lu", currentModel->CloudSize());
        ImGui::Text("Triangle count: %lu", currentModel->TriangleCount());
        ImGui::Text("Reconstruction time: %f", reconstructionTime);
        ImGui::Text("Normals estimated: %s", currentModel->m_NormalsEstimated ? "Yes" : "No");
        if (ImGui::Button("Estimate normals"))
            currentModel->EstimateNormals();

        ImGui::Combo("Method", &methodID, s_MethodLabels.data(), s_MethodLabels.size());

        auto method = static_cast<ReconstructionMethod>(methodID);
        if (ImGui::Button("Reconstruct"))
            reconstructionTime = currentModel->Reconstruct(method);

        auto renderGridGUI = [&]() {
            // Cannot merge due to short circuit || evaluation. Causes flickering
            // when moving the slider because the render command is skipped
            ImGui::Text("Grid resolution");
            if (ImGui::SliderInt("X", &gridResX, 5, 100))
                currentModel->m_Grid.SetResX(gridResX);
            if (ImGui::SliderInt("Y", &gridResY, 5, 100))
                currentModel->m_Grid.SetResY(gridResY);
            if (ImGui::SliderInt("Z", &gridResZ, 5, 100))
                currentModel->m_Grid.SetResZ(gridResZ);
            if (ImGui::Button("Regenerate grid"))
                currentModel->m_Grid.Regenerate();
        };

        switch (method) {
            case ReconstructionMethod::ModifiedHoppe:
                ImGui::Text("Neighbourhood size");
                if (ImGui::SliderInt("", &neighbourhoodSize, 1, 10))
                    currentModel->m_NeighbourhoodSize = (size_t)neighbourhoodSize;

                renderGridGUI();
                break;

            case ReconstructionMethod::MLS:
                ImGui::Text("Neighbourhood size");
                if (ImGui::SliderInt("", &neighbourhoodSize, 1, 25))
                    currentModel->m_NeighbourhoodSize = (size_t)neighbourhoodSize;

                ImGui::Checkbox("Use median filter", &currentModel->m_MLS_use_median);
                ImGui::SliderInt("Polynom degree", &currentModel->m_MLS_degree, 0, 3);
                renderGridGUI();
                break;

            case ReconstructionMethod::PCL_Hoppe:
                ImGui::SliderFloat("Iso level", &currentModel->m_IsoLevel, 0.0f, 1.0f);
                renderGridGUI();
                break;

            case ReconstructionMethod::PCL_MarchingCubesRBF:
                ImGui::Text("Off surface displacement");
                ImGui::SliderFloat("", &currentModel->m_OffSurfaceDisplacement, 0.0f, 1.0f);
                ImGui::SliderFloat("Iso level", &currentModel->m_IsoLevel, 0.0f, 1.0f);
                renderGridGUI();
                break;

            case ReconstructionMethod::PCL_Poisson:
                ImGui::SliderInt("Depth", &currentModel->m_Depth, 1, 10);
                ImGui::SliderInt("Minimum depth", &currentModel->m_MinDepth, 1, 10);
                ImGui::SliderInt("Iso divide", &currentModel->m_IsoDivide, 1, 16);
                ImGui::SliderInt("Solver divide", &currentModel->m_SolverDivide, 1, 16);
                ImGui::SliderFloat("Point weight", &currentModel->m_PointWeight, 1.0f, 10.0f);
                ImGui::SliderFloat("Samples per node", &currentModel->m_SamplesPerNode, 1.0f, 10.0f);
                ImGui::SliderFloat("Scale", &currentModel->m_Scale, 1.1f, 5.0f);
                break;

            case ReconstructionMethod::PCL_ConcaveHull:
                ImGui::SliderFloat("Alpha", &currentModel->m_Alpha, 0.00001f, 0.2f);
                break;

            case ReconstructionMethod::PCL_ConvexHull:
                break;

            case ReconstructionMethod::PCL_GreedyProjectionTriangulation:
                ImGui::SliderInt("Maximum NN", &currentModel->m_MaxNN, 1, 300);
                ImGui::SliderFloat("Max surface angle", &currentModel->m_MaxSurfaceAngle, 1.0f, 150.0f);
                ImGui::SliderFloat("Max triangle angle", &currentModel->m_MaxAngle, 1.0f, 150.0f);
                ImGui::SliderFloat("Min triangle angle", &currentModel->m_MinAngle, 1.0f, 120.0f);
                ImGui::SliderFloat("Search radius", &currentModel->m_SearchRadius, 0.1f, 10.0f);
                ImGui::SliderFloat("Mu", &currentModel->m_Mu, 0.1f, 10.0f);
                break;

            case ReconstructionMethod::PCL_OrganizedFastMesh:
                ImGui::SliderFloat("Angle tolerance", &currentModel->m_AngleTolerance, 1.0f, 90.0f);
                ImGui::SliderFloat("Distance tolerance", &currentModel->m_DistTolerance, 0.0f, 10.0f);
                ImGui::SliderFloat("Max edge length", &currentModel->m_A, 0.001f, 1.0f);
                break;
        }

        ImGui::Text("Debug Options");
        ImGui::Checkbox("Show mesh", &currentModel->m_ShowMesh);
        ImGui::Checkbox("Show grid", &currentModel->m_ShowGrid);
        ImGui::Checkbox("Show BB", &currentModel->m_ShowBB);
        ImGui::Checkbox("Show input PC", &currentModel->m_ShowInputPC);
        if (ImGui::Checkbox("Show normals", &currentModel->m_ShowNormals)) {
            currentModel->EstimateNormals();
        }
        if (ImGui::Checkbox("Show spanning tree", &currentModel->m_ShowSpanningTree)) {
            currentModel->EstimateNormals();
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
