#ifndef CLOUD_MODEL_H
#define CLOUD_MODEL_H

#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <tuple>
#include <map>
#include <BaseApp.h>
#include <Loader.h>
#include <Gui.h>

struct VertexRGB {
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 color = glm::vec3(0.0f);

    VertexRGB() = default;

    explicit VertexRGB(const glm::vec3 &pos, const glm::vec3 &color = glm::vec3(0.0f)) : pos(pos), color(color) {}
};

struct VertexNormal {
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);

    VertexNormal() = default;

    explicit VertexNormal(const glm::vec3 &pos, const glm::vec3 &normal = glm::vec3(0.0f)) : pos(pos), normal(normal) {}
};

struct VertexRGBNormal {
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 color = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);

    VertexRGBNormal() = default;

    explicit VertexRGBNormal(const glm::vec3 &pos, const glm::vec3 &color = glm::vec3(0.0f),
                             const glm::vec3 &normal = glm::vec3(0.0f)) : pos(pos), color(color), normal(normal) {}
};


class CloudModel {
    pcl::PointCloud<pcl::PointNormal>::Ptr m_Cloud;
    pcl::search::KdTree<pcl::PointNormal>::Ptr m_Tree;
    Eigen::Vector4f m_MinBB;
    Eigen::Vector4f m_MaxBB;
    Eigen::Vector4f m_SizeBB;
    std::vector<pcl::PointNormal> m_Connections;
    std::vector<VertexRGB> m_CubeCorners;
    std::vector<std::tuple<size_t, size_t, size_t>> m_CornersXYZ;
    std::vector<float> m_IsoValues;

    std::map<std::pair<size_t, size_t>, size_t> m_VertexIndices;
    std::vector<GLuint> m_MeshIndices;
    std::vector<glm::vec3> m_MeshVertices;
    std::vector<glm::vec3> m_CloudNormals;

    // Indices of MC edge vertices in order
    // compatible with the edge and triangle table.
    // Used when computing positions of edge intersections
    static constexpr std::array<size_t, 24> m_RelativeCornerIndices{
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            4, 5,
            5, 6,
            6, 7,
            7, 4,
            0, 4,
            1, 5,
            2, 6,
            3, 7
    };

    enum Buffers {
        Model,
        Corners,
        InputPC,
        InputPCNormals,
        Connections,
        BUFFER_COUNT
    };

    std::array<GLuint, BUFFER_COUNT> m_VAOs = {};
    std::array<GLuint, BUFFER_COUNT> m_VBOs = {};
    std::array<GLuint, BUFFER_COUNT> m_EBOs = {};

    static ProgramObject s_MeshProgram;
    static ProgramObject s_ShaderProgram;
    static ProgramObject s_GeometryProgram;
    static ProgramObject s_ColorProgram;

public:
    int m_GridSizeX = 10;
    int m_GridSizeY = 10;
    int m_GridSizeZ = 10;
    bool m_ShowGrid = true;
    bool m_InvalidatedGrid = false;
    int m_ConnectionIdx = 0;
    bool m_ShowMesh = false;
    bool m_ShowInputPC = false;
    bool m_ShowNormals = false;
    bool m_ShowConnections = false;

    size_t GetCornerCount() { return m_CubeCorners.size(); }

    CloudModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir);

    void Draw(glm::mat4 pv, glm::vec3 color);

    void RegenerateGrid();

    void CalculateIsoValues();

    void Reconstruct();
};


#endif