#ifndef CLOUD_MODEL_H
#define CLOUD_MODEL_H

#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

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


class CloudModel
{
    pcl::PointCloud<pcl::PointNormal>::Ptr m_Cloud;
    pcl::search::KdTree<pcl::PointNormal>::Ptr m_Tree;
    Eigen::Vector4f m_MinBB;
    Eigen::Vector4f m_MaxBB;
    Eigen::Vector4f m_SizeBB;
    std::vector<pcl::PointNormal> m_Connections;
    std::vector<BasicVertex> m_CubeCorners;
    std::vector<float> m_IsoValues;

    std::vector<glm::vec3> m_MeshVertices;

    enum Buffers
    {
        Model,
        Corners,
        InputPC,
        Connections,
        BUFFER_COUNT
    };

    std::array<GLuint, BUFFER_COUNT> m_VAOs;
    std::array<GLuint, BUFFER_COUNT> m_VBOs;
    std::array<GLuint, BUFFER_COUNT> m_EBOs;

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

    CloudModel(const std::string& shaderDir);

    void Draw(glm::mat4 pv, glm::vec3 color);

    void RegenerateGrid();

    void CalculateIsoValues();

    void Reconstruct();
};


#endif