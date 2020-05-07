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


struct Cube {
    // Indices of MC edge vertices in order
    // compatible with the edge and triangle table.
    // Used when computing positions of edge intersections
    static constexpr std::array<GLuint, 24> s_CornerIndices{
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
    std::array<glm::vec3, 8> m_Corners;

    glm::vec3 min;
    glm::vec3 max;
    glm::vec3 size;

    GLuint m_VAO = 0;
    GLuint m_VBO = 0;
    GLuint m_EBO = 0;

    BoundingBox(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, float increase) {
        Eigen::Vector4f minTmp, maxTmp;
        pcl::getMinMax3D(*cloud, minTmp, maxTmp);
        min = glm::vec3(minTmp.x(), minTmp.y(), minTmp.z());
        max = glm::vec3(maxTmp.x(), maxTmp.y(), maxTmp.z());
        size = max - min;
        min -= (size * (increase / 2.0f));
        max += (size * (increase / 2.0f));
        size *= (1.0f + increase);

        m_Corners = {
            min,
            min + glm::vec3(size.x, 0.0f, 0.0f),
            min + glm::vec3(size.x, size.y, 0.0f),
            min + glm::vec3(0.0f, size.y, 0.0f),

            min + glm::vec3(0.0f, 0.0f, size.z),
            min + glm::vec3(size.x, 0.0f, size.z),
            max,
            min + glm::vec3(0.0f, size.y, size.z),
        };


        glCreateVertexArrays(1, &m_VAO);
        glCreateBuffers(1, &m_VBO);
        glCreateBuffers(1, &m_EBO);

        glVertexArrayElementBuffer(m_VAO, m_EBO);
        glEnableVertexArrayAttrib(m_VAO, 0);
        glVertexArrayAttribFormat(m_VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(m_VAO, 0, m_VBO, 0, sizeof(glm::vec3));
        glNamedBufferData(m_VBO, sizeof(glm::vec3) * m_Corners.size(), m_Corners.data(), GL_STATIC_DRAW);
        glNamedBufferData(m_EBO, sizeof(GLuint) * Cube::s_CornerIndices.size(), Cube::s_CornerIndices.data(), GL_STATIC_DRAW);
    }

    void Draw(ProgramObject &shader, glm::mat4 pvm) const;

    void Print() {
        printf("[BB] Size: [%f, %f, %f]\n", size.x, size.y, size.z);
        printf("[BB] Min: [%f, %f, %f]\n", min.x, min.y, min.z);
        printf("[BB] Max: [%f, %f, %f]\n", max.x, max.y, max.z);
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
    PCL_MarchingCubesRBF,
    PCL_Poisson,
    PCL_ConcaveHull,
    PCL_ConvexHull,
    PCL_GreedyProjectionTriangulation,
    PCL_OrganizedFastMesh
};

static std::array<const char *, 9> s_MethodLabels{
        "Modified Hoppe's",
        "MLS",
        "PCL: Hoppe's",
        "PCL: Marching Cubes RBF",
        "PCL: Poisson",
        "PCL: ConcaveHull (alpha shapes)",
        "PCL: ConvexHull",
        "PCL: Greedy Projection Triangulation",
        "PCL: Organized Fast Mesh"
};


class CloudModel {
    friend class Grid;

    enum Buffers {
        Mesh,
        Cloud,
        CloudNormals,
        SpanningTree,
        BUFFER_COUNT
    };

    std::string m_Name;


    pcl::PointCloud<pcl::PointNormal>::Ptr m_Cloud;
    pcl::search::KdTree<pcl::PointNormal>::Ptr m_Tree;

    std::vector<GLuint> m_MeshIndices;
    std::vector<glm::vec3> m_MeshVertices;
    std::vector<glm::vec3> m_CloudNormals;
    std::vector<glm::vec3> m_TreeEdges; // Spanning tree edges for visualization

    /* Maps indices of two vertices forming an MC edge to an index of an interpolated edge vertex */
    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> m_VertexIndices;

    std::array<GLuint, BUFFER_COUNT> m_VAOs = {};
    std::array<GLuint, BUFFER_COUNT> m_VBOs = {};
    std::array<GLuint, BUFFER_COUNT> m_EBOs = {};

    static ProgramObject s_MeshShader;
    static ProgramObject s_ShaderProgram;
    static ProgramObject s_GeometryProgram;
    static ProgramObject s_ColorProgram;

    void BufferData();

    void ExtractPclReconstructionData(const std::vector<pcl::Vertices> &outputIndices,
                                      const pcl::PointCloud<pcl::PointNormal>::Ptr &surfacePoints);

    void MarchCubes();

    double HoppeReconstruction();

    double MLSReconstruction();

    double PCL_HoppeReconstruction();

    double PCL_MC_RBF_Reconstruction();

    double PCL_PoissonReconstruction();

    double PCL_ConcaveHullReconstruction();

    double PCL_ConvexHullReconstruction();

    double PCL_GP3();

    double PCL_OrganizedFastMeshReconstruction();

public:
    BoundingBox m_BB;
    Grid m_Grid;
    float m_IsoLevel = 0.0f;
    size_t m_NeighbourhoodSize = 1;

    int m_Depth = 8;
    int m_MinDepth = 5;
    int m_IsoDivide = 8;
    int m_SolverDivide = 8;
    float m_PointWeight = 4.0f;
    float m_SamplesPerNode = 1.0f;
    float m_Scale = 1.1f;
    float m_OffSurfaceDisplacement = 0.0f;

    // MLS
    int m_MLS_degree = 2;
    bool m_MLS_use_median = false;

    float m_Alpha = 1.0f;

    // GP3
    float m_SearchRadius = 0.1f;
    float m_Mu = 0.1f;
    float m_MaxAngle = 120.0f;
    float m_MinAngle = 10.0f;
    float m_MaxSurfaceAngle = 45.0f;
    int m_MaxNN = 100;

    // FastMesh
    float m_AngleTolerance = 12.5f;
    float m_DistTolerance = 0;
    float m_A = 0.15f;
    float m_B = 0.0f;
    float m_C = 0.0f;

    bool m_ShowBB = true;
    bool m_ShowGrid = false;
    bool m_ShowMesh = true;
    bool m_ShowInputPC = true;
    bool m_ShowNormals = false;
    bool m_ShowSpanningTree = false;
    bool m_NormalsEstimated = false;

    CloudModel(const std::string& name, pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir);

    const std::string& Name() const { return m_Name; }

    void Draw(glm::mat4 pv, glm::vec3 color) const;

    double Reconstruct(ReconstructionMethod method);

    size_t CloudSize() const { return m_Cloud->points.size(); }

    size_t TriangleCount() const { return m_MeshIndices.size() / 3; }

    void EstimateNormals();
};


#endif