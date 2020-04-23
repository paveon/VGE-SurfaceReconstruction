#define SDL_MAIN_HANDLED

#include <iostream>
#include <thread>
#include <chrono>

//#include <pcl/common/common.h>
//#include <pcl/common/vector_average.h>
//#include <pcl/Vertices.h>

//#include <pcl/surface/3rdparty/poisson4/octree_poisson.h>
//#include <pcl/surface/3rdparty/poisson4/sparse_matrix.h>
//#include <pcl/surface/3rdparty/poisson4/function_data.h>
//#include <pcl/surface/3rdparty/poisson4/ppolynomial.h>
//#include <pcl/surface/3rdparty/poisson4/multi_grid_octree_data.h>
//#include <pcl/surface/3rdparty/poisson4/geometry.h>

#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>

//#include <pcl/surface/poisson.h>
#include "poisson2.h"

#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>
#include "bunny.h"

struct BasicVertex
{
    glm::vec3 pos;
    glm::vec3 color;

    BasicVertex() = default;

    BasicVertex(const glm::vec3 &pos, const glm::vec3 &color) : pos(pos), color(color) {}
};

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

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithNormals()
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals(new pcl::PointCloud<pcl::PointNormal>());
    for (const auto &bunnyVertice : bunnyVertices)
    {
        auto *pt = new pcl::PointNormal;
        pt->x = bunnyVertice.position[0];
        pt->y = bunnyVertice.position[1];
        pt->z = bunnyVertice.position[2];
        pt->normal_x = bunnyVertice.normal[0];
        pt->normal_y = bunnyVertice.normal[1];
        pt->normal_z = bunnyVertice.normal[2];
        cloud_point_normals->push_back(*pt);
    }
    return cloud_point_normals;
}

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithOutNormals()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto &bunnyVertice : bunnyVertices)
    {
        auto *pt = new pcl::PointXYZ;
        pt->x = bunnyVertice.position[0];
        pt->y = bunnyVertice.position[1];
        pt->z = bunnyVertice.position[2];
        cloud_point->push_back(*pt);
    }
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud_point);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimation.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    // Use all neighbors in a sphere of radius 3cm
    normalEstimation.setRadiusSearch(0.03);
    // Compute the features
    normalEstimation.compute(*cloud_normals);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud_point, *cloud_normals, *cloud_point_normals);
    std::cout << cloud_point_normals->size() << std::endl;
    return cloud_point_normals;
}

void show(const pcl::PolygonMesh &mesh)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setCameraPosition(0, 0, 0, 0, 0, 0);
    viewer->addPolygonMesh(mesh);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> *ne = new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>;
    ne->setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne->setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne->setRadiusSearch(0.3);

    // Compute the features
    ne->compute(*cloud_normals);
    return cloud_normals;
}

int main(int /*argc*/, char ** /*argv*/)
{
    std::vector<GLuint> flatIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());

    // Simple caching system for later (we might want to display multiple models during presentation?)
    std::ifstream indexCache("index_cache.dat", ios::in | ios::binary);
    if (pcl::io::loadPCDFile<pcl::PointNormal>("surface_cache.pcd", *surfaceCloud) >= 0 && indexCache.is_open())
    {
        // Cache exists
        size_t cacheSize;
        indexCache.read((char *)&cacheSize, sizeof(size_t));
        flatIndices.resize(cacheSize);
        indexCache.read((char *)flatIndices.data(), cacheSize * sizeof(GLuint));
    }
    else
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals;
        if (false)
            cloud_point_normals = bunnyWithOutNormals();
        else
            cloud_point_normals = bunnyWithNormals();

        pcl::Poisson<pcl::PointNormal> poisson;
        poisson.setDepth(6); // Was painfully slow with depth of 12
        poisson.setInputCloud(cloud_point_normals);

        // Reconstruction
        std::vector<pcl::Vertices> outputIndices;
        poisson.reconstruct(*surfaceCloud, outputIndices);
        //    pcl::PolygonMesh mesh;
        //    poisson.reconstruct(mesh)
        
        // PCL uses weird memory layout for indices, copy the data
        // into new vector with flat layout so that it can be used by OpenGL
        size_t idx = 0;
        flatIndices.resize(outputIndices.size() * 3);
        for (size_t i = 0; i < outputIndices.size(); i++)
        {
            const auto &indices = outputIndices[i];
            for (size_t index : indices.vertices)
                flatIndices[idx++] = index;
        }

        // Save cache
        pcl::io::savePCDFileASCII("surface_cache.pcd", *surfaceCloud);
        std::ofstream cacheFile("index_cache.dat", ios::out | ios::binary);
        size_t cacheSize = flatIndices.size();
        cacheFile.write((char *)&cacheSize, sizeof(size_t));
        cacheFile.write((char *)flatIndices.data(), cacheSize * sizeof(GLuint));
        cacheFile.close();
    }

    auto &surfacePoints = surfaceCloud->points;
    std::cout << "Cloud size: " << surfacePoints.size() << std::endl;
    std::cout << "Triangle count: " << (flatIndices.size() / 3) << std::endl;

    // show(mesh);

    BaseApp app;
    PerspectiveCamera cam;
    OrbitManipulator manipulator(&cam);
    manipulator.setZoom(5.0f);
    manipulator.setRotationX(-90.0f);
    manipulator.setupCallbacks(app);

    g_ShaderFolder = app.getResourceDir() + "Shaders/";

    GLuint vaoCoord, vboCoord;
    GLuint rayVAO, rayVBO;
    GLuint modelVAO, modelVBO, modelEBO;
    ProgramObject wireframeProgram;
    ProgramObject basicProgram;

    auto mainWindow = app.getMainWindow();

    bool showErrorPopup = false;
    bool showDemo = true;
    bool rayCast = false;

    ImVec2 optSize(50, 100);
    ImVec2 buttonSize(120, 20);

    char errorBuffer[256];

    app.addInitCallback([&]() {
        auto wireframeVS = compileShader(GL_VERTEX_SHADER, Loader::text(g_ShaderFolder + "wireframe.vert"));
        auto wireframeFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(g_ShaderFolder + "wireframe.frag"));
        wireframeProgram = createProgram(wireframeVS, wireframeFS);

        auto basicVS = compileShader(GL_VERTEX_SHADER, Loader::text(g_ShaderFolder + "primitive.vert"));
        auto basicFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(g_ShaderFolder + "primitive.frag"));
        basicProgram = createProgram(basicVS, basicFS);

        // Buffers for ray cast visualization
        glCreateVertexArrays(1, &rayVAO);
        glCreateBuffers(1, &rayVBO);
        glNamedBufferData(rayVBO, sizeof(glm::vec3) * 2 + 4, 0, GL_DYNAMIC_DRAW);
        glVertexArrayVertexBuffer(rayVAO, 0, rayVBO, offsetof(BasicVertex, pos), sizeof(glm::vec3));
        glEnableVertexArrayAttrib(rayVAO, 0);

        // Buffers for coordinate system arrows
        glCreateVertexArrays(1, &vaoCoord);
        glCreateBuffers(1, &vboCoord);
        glNamedBufferData(vboCoord, sizeof(BasicVertex) * g_CoordVertices.size() + 4, g_CoordVertices.data(), GL_STATIC_DRAW);
        glVertexArrayVertexBuffer(vaoCoord, 0, vboCoord, offsetof(BasicVertex, pos), sizeof(BasicVertex));
        glVertexArrayVertexBuffer(vaoCoord, 1, vboCoord, offsetof(BasicVertex, color), sizeof(BasicVertex));
        glEnableVertexArrayAttrib(vaoCoord, 0);
        glEnableVertexArrayAttrib(vaoCoord, 1);


        // Buffer model vertex and index data and bind buffers to VAO
        glCreateVertexArrays(1, &modelVAO);
        glCreateBuffers(1, &modelVBO);
        glCreateBuffers(1, &modelEBO);
        glNamedBufferData(modelVBO, sizeof(pcl::PointNormal) * surfacePoints.size() + 4, surfacePoints.data(), GL_STATIC_DRAW);
        glNamedBufferData(modelEBO, sizeof(GLuint) * flatIndices.size(), flatIndices.data(), GL_STATIC_DRAW);
        glVertexArrayVertexBuffer(modelVAO, 0, modelVBO, offsetof(pcl::PointNormal, data), sizeof(pcl::PointNormal));
        glVertexArrayVertexBuffer(modelVAO, 1, modelVBO, offsetof(pcl::PointNormal, normal), sizeof(pcl::PointNormal));
        // glVertexArrayVertexBuffer(modelVAO, 2, modelVBO, offsetof(pcl::PointNormal, curvature), sizeof(pcl::PointNormal));
        glEnableVertexArrayAttrib(modelVAO, 0);
        glEnableVertexArrayAttrib(modelVAO, 1);
        // glEnableVertexArrayAttrib(modelVAO, 2);
        glVertexArrayElementBuffer(modelVAO, modelEBO);

        ImGui::GetIO().WantTextInput = true;
        ImGui::GetIO().WantCaptureKeyboard = true;
    });

    glm::vec3 color(0.0, 1.0, 0.0);
    glm::vec3 wireframeColor(1.0, 0.0, 0.0);

    app.addDrawCallback([&]() {
        int w = mainWindow->getWidth();
        int h = mainWindow->getHeight();
        glViewport(0, 0, w, h);
        glClearColor(0.2, 0.2, 0.2, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Draw coordinate arrows
        glm::mat4 pvm = cam.getProjection() * cam.getView();
        wireframeProgram.use();
        wireframeProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(vaoCoord);
        glLineWidth(3.0f);
        glDrawArrays(GL_LINES, 0, g_CoordVertices.size());
        glLineWidth(1.0f);

        if (rayCast)
        {
            glBindVertexArray(rayVAO);
            glDrawArrays(GL_LINES, 0, 2);
        }

        // Draw model
        basicProgram.use();
        basicProgram.set3fv("primitiveColor", glm::value_ptr(color));
        basicProgram.set1ui("textureType", 0);
        basicProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(modelVAO);
        glDrawElements(GL_TRIANGLES, flatIndices.size(), GL_UNSIGNED_INT, nullptr);

        // Draw outlines of model triangles, it looks weird without illumination model otherwise
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        basicProgram.set3fv("primitiveColor", glm::value_ptr(wireframeColor));
        glDrawElements(GL_TRIANGLES, flatIndices.size(), GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // GUI
        label("FPS: " + std::to_string(ImGui::GetIO().Framerate));
        ImGui::Begin("Options", nullptr, optSize);
        ImGui::Text("TODO GUI");
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

    app.addMousePressCallback([&](uint8_t button, int x, int y) {
        if (button == 3)
        {
            std::array<glm::vec3, 2> rayVertices{rayOrigin, rayOrigin + (rayDir * 100.0f)};
            glNamedBufferSubData(rayVBO, 0, sizeof(glm::vec3) * 2, rayVertices.data());
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
