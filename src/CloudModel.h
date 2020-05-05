#ifndef CLOUD_MODEL_H
#define CLOUD_MODEL_H

#if defined(_OPENMP)

#include <omp.h>

#else
void omp_set_num_threads (int);
int omp_get_num_threads();
int omp_get_max_threads();
int omp_get_thread_num();
int omp_get_num_procs();
#endif

#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <tuple>
#include <unordered_map>
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


struct BoundingBox {
    Eigen::Vector4f min;
    Eigen::Vector4f max;
    Eigen::Vector4f size;

    BoundingBox(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, float increase) {
        pcl::getMinMax3D(*cloud, min, max);
        size = max - min;
        min -= (size * (increase / 2.0f));
        max += (size * (increase / 2.0f));
        size *= (1.0f + increase);
    }

    void Print() {
        printf("[BB] Size: [%f, %f, %f]\n", size.x(), size.y(), size.z());
        printf("[BB] Min: [%f, %f, %f]\n", min.x(), min.y(), min.z());
        printf("[BB] Max: [%f, %f, %f]\n", max.x(), max.y(), max.z());
    }
};

class CloudModel;

class Grid {
    CloudModel &m_ParentModel;
    GLuint m_VAO = 0;
    GLuint m_VBO = 0;

    size_t m_ResX = 10;
    size_t m_ResY = 10;
    size_t m_ResZ = 10;
    bool m_Invalidated = true;

public:
    std::vector<VertexRGB> m_Points; /* Grid points */
    std::vector<float> m_IsoValues; /* Corresponding iso values */

    Grid(CloudModel &model) : m_ParentModel(model) {
        glCreateVertexArrays(1, &m_VAO);
        glCreateBuffers(1, &m_VBO);

        // Setup vertex attributes for MC corner data
        glEnableVertexArrayAttrib(m_VAO, 0);
        glEnableVertexArrayAttrib(m_VAO, 1);
        glVertexArrayAttribFormat(m_VAO, 0, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, pos));
        glVertexArrayAttribFormat(m_VAO, 1, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, color));
        glVertexArrayVertexBuffer(m_VAO, 0, m_VBO, 0, sizeof(VertexRGB));
        glVertexArrayVertexBuffer(m_VAO, 1, m_VBO, 0, sizeof(VertexRGB));

        Regenerate();
    }

    void Regenerate();

    void CalculateIsoValues(size_t neighbourhoodSize);

    void CalculateIsoValuesMLS(size_t neighbourhoodSize);

    void Draw(ProgramObject &shader, glm::mat4 pvm) const;

    size_t GetResX() const { return m_ResX; }

    size_t GetResY() const { return m_ResX; }

    size_t GetResZ() const { return m_ResX; }

    void SetResX(size_t value) {
        m_ResX = value;
        m_Invalidated = true;
    }

    void SetResY(size_t value) {
        m_ResY = value;
        m_Invalidated = true;
    }

    void SetResZ(size_t value) {
        m_ResZ = value;
        m_Invalidated = true;
    }
};

struct MarchingCube {
    // Indices of MC edge vertices in order
    // compatible with the edge and triangle table.
    // Used when computing positions of edge intersections
    static constexpr std::array<size_t, 24> s_CornerIndices{
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
};


struct HoppeParameters {
};


struct PairHash {
public:
    template<typename T, typename U>
    std::size_t operator()(const std::pair<T, U> &x) const {
        return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
    }
};


enum class ReconstructionMethod {
    ModifiedHoppe,
    MLS,
    PCL_Hoppe,
    PCL_Poisson
};


class CloudModel {
    friend class Grid;

    pcl::PointCloud<pcl::PointNormal>::Ptr m_Cloud;
    pcl::search::KdTree<pcl::PointNormal>::Ptr m_Tree;

    std::vector<GLuint> m_MeshIndices;
    std::vector<glm::vec3> m_MeshVertices;
    std::vector<glm::vec3> m_CloudNormals;

    /* Maps indices of two vertices forming an MC edge to an index of an interpolated edge vertex */
    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> m_VertexIndices;

    enum Buffers {
        Mesh,
        Cloud,
        CloudNormals,
        BUFFER_COUNT
    };

    std::array<GLuint, BUFFER_COUNT> m_VAOs = {};
    std::array<GLuint, BUFFER_COUNT> m_VBOs = {};
    std::array<GLuint, BUFFER_COUNT> m_EBOs = {};

    static ProgramObject s_MeshProgram;
    static ProgramObject s_ShaderProgram;
    static ProgramObject s_GeometryProgram;
    static ProgramObject s_ColorProgram;

    void BufferData();

    void ExtractPclReconstructionData(const std::vector<pcl::Vertices> &outputIndices,
                                      const pcl::PointCloud<pcl::PointNormal>::Ptr &surfacePoints);

    void HoppeReconstruction();

    void PCL_HoppeReconstruction();

    void MLSReconstruction();

    void PCL_PoissonReconstruction();

public:
    BoundingBox m_BB;
    Grid m_Grid;
    float m_IsoLevel = 0.0f;
    float m_IgnoreDistance = -1.0f;
    size_t m_NeighbourhoodSize = 1;

    int m_Degree = 2;
    int m_Depth = 8;
    int m_MinDepth = 5;
    int m_IsoDivide = 8;
    int m_SolverDivide = 8;
    float m_PointWeight = 4.0f;
    float m_SamplesPerNode = 1.0f;
    float m_Scale = 1.1f;

    bool m_ShowGrid = true;
    bool m_ShowMesh = true;
    bool m_ShowInputPC = false;
    bool m_ShowNormals = false;

    CloudModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir);

    void Draw(glm::mat4 pv, glm::vec3 color);

    void Reconstruct(ReconstructionMethod method);
};


#endif