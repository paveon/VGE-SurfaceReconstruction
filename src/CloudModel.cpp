#include "CloudModel.h"
#include "MCTable.h"

#include <chrono>
#include <pcl/common/pca.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/poisson.h>
#include <Eigen/Dense>
#include "MLS_helper_functions.cpp"

#if defined(_OPENMP)

#include <omp.h>

#else

void omp_set_num_threads(int) {}

int omp_get_num_threads() { return 1; }

int omp_get_max_threads() { return 1; }

int omp_get_thread_num() { return 0; }

int omp_get_num_procs() { return 1; }

#endif

class ClockGuard {
    std::chrono::time_point<std::chrono::steady_clock> m_Start;
    const char* m_Name;

public:
    ClockGuard(const char* name) : m_Start(std::chrono::steady_clock::now()), m_Name(name) {}

    ~ClockGuard() {
        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - m_Start;
        std::cout << "[Timer] " << m_Name << ": " << elapsed_seconds.count() << "s" << std::endl;
    }
};


void Grid::Regenerate() {
    if (!m_Invalidated)
        return;

    ClockGuard timer(__func__);

    const BoundingBox bb(m_ParentModel.m_BB);
    const float deltaX = bb.size.x() / (float) m_ResX;
    const float deltaY = bb.size.y() / (float) m_ResY;
    const float deltaZ = bb.size.z() / (float) m_ResZ;
    const size_t yzPointCount = (m_ResZ + 1) * (m_ResY + 1);
    const size_t pointsTotal = (m_ResX + 1) * yzPointCount;
    bool glBufferOverflow = pointsTotal > m_Points.size();

    m_Points.resize(pointsTotal);
    m_IsoValues.resize(pointsTotal);

#pragma omp parallel for schedule(static) firstprivate(deltaX, deltaY, deltaZ, bb, yzPointCount)
    for (size_t x = 0; x < m_ResX + 1; ++x) {
        size_t gridIdx = x * yzPointCount;
        float xPos = bb.min.x() + (x * deltaX);

        for (size_t y = 0; y < m_ResY + 1; ++y) {
            float yPos = bb.min.y() + (y * deltaY);

            for (size_t z = 0; z < m_ResZ + 1; ++z) {
                float zPos = bb.min.z() + (z * deltaZ);
                m_Points[gridIdx++] = VertexRGB(glm::vec3(xPos, yPos, zPos), glm::vec3(0.0f, 0.0f, 1.0f));
            }
        }
    }

    // Buffer grid points
    if (glBufferOverflow)
        glNamedBufferData(m_VBO, sizeof(VertexRGB) * m_Points.size(), m_Points.data(), GL_STATIC_DRAW);
    else
        glNamedBufferSubData(m_VBO, 0, sizeof(VertexRGB) * m_Points.size(), m_Points.data());

    m_Invalidated = false;
}


void Grid::CalculateIsoValues(size_t neighbourhoodSize) {
    ClockGuard timer(__func__);

#pragma omp parallel
    {
        pcl::IndicesPtr indices(new std::vector<int>);
        indices->resize(neighbourhoodSize);
        std::vector<float> distances(neighbourhoodSize);

        int threadCount = omp_get_num_threads();
        int threadID = omp_get_thread_num();
        size_t chunkSize = m_Points.size() / threadCount;
        size_t startIdx = chunkSize * threadID;
        size_t endIdx = chunkSize * (threadID + 1);
        if (threadID == threadCount - 1)
            endIdx = m_Points.size();

        for (size_t i = startIdx; i < endIdx; ++i) {
            VertexRGB &gridVertex(m_Points[i]);
            pcl::PointNormal gridPoint;
            std::memcpy(gridPoint.data, glm::value_ptr(gridVertex.pos), sizeof(float) * 3);

            int found = m_ParentModel.m_Tree->nearestKSearch(gridPoint, neighbourhoodSize, *indices, distances);
            assert((size_t) found == neighbourhoodSize);

            float distance = 0.0f;
            if (neighbourhoodSize > 1) {
                pcl::PCA<pcl::PointNormal> pca;
                pca.setInputCloud(m_ParentModel.m_Cloud);
                pca.setIndices(indices);
                auto centroid = pca.getMean();
                auto eigenVectors = pca.getEigenVectors();
                glm::vec3 centroidVertex(centroid.x(), centroid.y(), centroid.z());
                pcl::PointNormal nearestPoint = m_ParentModel.m_Cloud->points[indices->front()];

                // Find eigen vector with lowest angle
                float max = 0;
                glm::vec3 normal;
                for (size_t vectorIdx = 0; vectorIdx < 3; vectorIdx++) {
                    auto eigenVector(eigenVectors.col(vectorIdx));
                    eigenVector.normalize();

                    float dotProduct = nearestPoint.getNormalVector3fMap().dot(eigenVector);
                    if (std::abs(dotProduct) >= max) {
                        max = dotProduct;

                        // Resulting normal (eigen vector) might be incorrectly oriented. Original
                        // paper deals with this by using complex algorithm which traverses Riemannian
                        // graph and consistently orients normals of connected tangent planes.
                        // We'll use a hack and orient the normal based on the orientation of the closest point
                        normal = glm::vec3(eigenVector.x(), eigenVector.y(), eigenVector.z());
                        if (max < 0) {
                            max = -max;
                            normal = -normal;
                        }
                    }
                }

                glm::vec3 direction(gridVertex.pos - centroidVertex);
                distance = glm::dot(direction, glm::normalize(normal));
            } else {
                pcl::PointNormal nearestPoint = m_ParentModel.m_Cloud->points[indices->front()];

                // Calculate the distance between the MC corner point and the tangent
                // plane of the closest surface point. Dot projection of the point2point
                // vector and the unit-length surface normal gives us the distance.
                auto direction = gridPoint.getVector3fMap() - nearestPoint.getVector3fMap();
                distance = nearestPoint.getNormalVector3fMap().dot(direction);
            }

            m_IsoValues[i] = distance;

            // Red color for corner points outside the geometry and green for points that are inside
            gridVertex.color = distance <= 0 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        }
    }

    // Update grid vertex data with colors
    glNamedBufferSubData(m_VBO, 0, sizeof(VertexRGB) * m_Points.size(), m_Points.data());
}


void Grid::Draw(ProgramObject &shader, glm::mat4 pvm) const {
    // Draw grid points
    shader.use();
    shader.setMatrix4fv("pvm", glm::value_ptr(pvm));
    glPointSize(3.0f);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, m_Points.size());
    glPointSize(1.0f);
}


CloudModel::CloudModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir) :
        m_Cloud(std::move(cloud)),
        m_Tree(new pcl::search::KdTree<pcl::PointNormal>()),
        m_BB(m_Cloud, 0.1f),
        m_Grid(*this) {

    m_Tree->setInputCloud(m_Cloud);
    m_BB.Print();

    if (!s_ShaderProgram) {
        auto basicVS = compileShader(GL_VERTEX_SHADER, Loader::text(shaderDir + "primitive.vert"));
        auto basicFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderDir + "primitive.frag"));
        auto normalsGS = compileShader(GL_GEOMETRY_SHADER, Loader::text(shaderDir + "primitive.geom"));
        s_ShaderProgram = createProgram(basicVS, basicFS);
        s_GeometryProgram = createProgram(basicVS, basicFS, normalsGS);

        auto colorVS = compileShader(GL_VERTEX_SHADER, Loader::text(shaderDir + "wireframe.vert"));
        auto colorFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderDir + "wireframe.frag"));
        s_ColorProgram = createProgram(colorVS, colorFS);

        auto meshVS = compileShader(GL_VERTEX_SHADER, Loader::text(shaderDir + "mesh.vert"));
        auto meshFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderDir + "mesh.frag"));
        s_MeshProgram = createProgram(meshVS, meshFS);
    }

    glCreateVertexArrays(m_VAOs.size(), m_VAOs.data());
    glCreateBuffers(m_VBOs.size(), m_VBOs.data());
    glCreateBuffers(m_EBOs.size(), m_EBOs.data());

    // Buffer point cloud data and setup vertex attributes
    m_CloudNormals.resize(m_Cloud->points.size() * 2);
    for (size_t i = 0; i < m_Cloud->points.size(); i += 2) {
        const auto &cloudPt = m_Cloud->points[i];
        m_CloudNormals[i] = glm::vec3(cloudPt.x, cloudPt.y, cloudPt.z);
        m_CloudNormals[i + 1] = glm::vec3(
                cloudPt.x + cloudPt.normal_x * 0.2f,
                cloudPt.y + cloudPt.normal_y * 0.2f,
                cloudPt.z + cloudPt.normal_z * 0.2f
        );
    }
    glNamedBufferData(m_VBOs[CloudNormals], sizeof(glm::vec3) * m_CloudNormals.size(), m_CloudNormals.data(),
                      GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[CloudNormals], 0);
    glVertexArrayAttribFormat(m_VAOs[CloudNormals], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[CloudNormals], 0, m_VBOs[CloudNormals], 0, sizeof(glm::vec3));

    glNamedBufferData(m_VBOs[Cloud], sizeof(pcl::PointNormal) * m_Cloud->points.size(), m_Cloud->points.data(),
                      GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[Cloud], 0);
    glEnableVertexArrayAttrib(m_VAOs[Cloud], 1);
    glVertexArrayAttribFormat(m_VAOs[Cloud], 0, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, data));
    glVertexArrayAttribFormat(m_VAOs[Cloud], 1, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, normal));
    glVertexArrayVertexBuffer(m_VAOs[Cloud], 0, m_VBOs[Cloud], 0, sizeof(pcl::PointNormal));
    glVertexArrayVertexBuffer(m_VAOs[Cloud], 1, m_VBOs[Cloud], 0, sizeof(pcl::PointNormal));

    m_Grid.Regenerate();
}

void CloudModel::Draw(glm::mat4 pv, glm::vec3 color) {
    glm::mat4 modelMatrix(1.0f);
    glm::mat4 pvm = pv * modelMatrix;
    glm::vec3 outlineColor(1.0f, 0.0f, 0.0f);

    if (m_ShowMesh && !m_MeshVertices.empty()) {
        // Draw model
        s_MeshProgram.use();
        s_MeshProgram.set3fv("primitiveColor", glm::value_ptr(color));
        s_MeshProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(m_VAOs[Mesh]);
        glDrawElements(GL_TRIANGLES, m_MeshIndices.size(), GL_UNSIGNED_INT, nullptr);

        // Draw outlines of model triangles, it looks weird without illumination model otherwise
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        s_MeshProgram.set3fv("primitiveColor", glm::value_ptr(outlineColor));
        glDrawElements(GL_TRIANGLES, m_MeshIndices.size(), GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    s_ShaderProgram.use();
    s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
    s_ShaderProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));

    if (m_ShowInputPC) {
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
        glPointSize(3.0f);
        glBindVertexArray(m_VAOs[Cloud]);
        glDrawArrays(GL_POINTS, 0, m_Cloud->points.size());
        glPointSize(1.0f);

        if (m_ShowNormals) {
            // Draw normals
            glm::vec3 normalColor(1.0f, 0.0f, 1.0f);

            // Version without geometry shader
//            s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(normalColor));
//            glLineWidth(2.0f);
//            glBindVertexArray(m_VAOs[InputPCNormals]);
//            glDrawArrays(GL_LINES, 0, m_CloudNormals.size());
//            glLineWidth(1.0f);

            // Geometry shader version
            s_GeometryProgram.use();
            s_GeometryProgram.set3fv("primitiveColor", glm::value_ptr(normalColor));
            s_GeometryProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
            glLineWidth(2.0f);
            glBindVertexArray(m_VAOs[Cloud]);
            glDrawArrays(GL_POINTS, 0, m_Cloud->points.size());
            glLineWidth(1.0f);
        }
    }

    if (m_ShowGrid) {
        m_Grid.Draw(s_ColorProgram, pvm);
    }
}


void CloudModel::Reconstruct(ReconstructionMethod method) {
    switch (method) {
        case ReconstructionMethod::ModifiedHoppe:
            HoppeReconstruction();
            break;
        case ReconstructionMethod::MLS:
            MLSReconstruction();
            break;

        case ReconstructionMethod::PCL_Hoppe:
            PCL_HoppeReconstruction();
            break;

        case ReconstructionMethod::PCL_Poisson:
            PCL_PoissonReconstruction();
            break;
    }
}


void CloudModel::HoppeReconstruction() {
    ClockGuard timer(__func__);

    m_Grid.Regenerate();
    m_Grid.CalculateIsoValues(m_NeighbourhoodSize);

    m_VertexIndices.clear();
    m_VertexIndices.rehash(m_Grid.m_Points.size() * 1.25f);
    m_MeshVertices.clear();
    m_MeshIndices.clear();

    size_t zPointCount = m_Grid.GetResZ() + 1;
    size_t yzPointCount = (m_Grid.GetResY() + 1) * zPointCount;

    std::array<GLuint, 12> intersections{};
    std::array<glm::vec3, 8> cubeCorners{};
    std::array<float, 8> isoValues{};

#pragma omp parallel for schedule(static) firstprivate(intersections, cubeCorners, isoValues, zPointCount, yzPointCount) shared(edgeTable, triangleTable)
    for (size_t x = 0; x < m_Grid.GetResX(); ++x) {
        for (size_t y = 0; y < m_Grid.GetResY(); ++y) {
            for (size_t z = 0; z < m_Grid.GetResZ(); ++z) {
                std::array<size_t, 8> gridCornerIndices{
                        z + (y * zPointCount) + (x * yzPointCount),
                        1 + z + (y * zPointCount) + (x * yzPointCount),
                        1 + z + ((y + 1) * zPointCount) + (x * yzPointCount),
                        z + ((y + 1) * zPointCount) + (x * yzPointCount),

                        z + (y * zPointCount) + ((x + 1) * yzPointCount),
                        1 + z + (y * zPointCount) + ((x + 1) * yzPointCount),
                        1 + z + ((y + 1) * zPointCount) + ((x + 1) * yzPointCount),
                        z + ((y + 1) * zPointCount) + ((x + 1) * yzPointCount)
                };

                for (size_t i = 0; i < cubeCorners.size(); ++i) {
                    cubeCorners[i] = m_Grid.m_Points[gridCornerIndices[i]].pos;
                    isoValues[i] = m_Grid.m_IsoValues[gridCornerIndices[i]];
                }

                uint32_t cubeIdx = 0;
                for (size_t i = 0; i < cubeCorners.size(); ++i)
                    cubeIdx |= unsigned(isoValues[i] < 0) << i;

                /* Cube is entirely in/out of the surface */
                if (cubeIdx == 0 || cubeIdx == 255)
                    continue;

                // Find the vertices where the surface intersects the cube
                uint32_t cubeConfig = edgeTable[cubeIdx];
                for (size_t i = 0; i < intersections.size(); ++i) {
                    if (cubeConfig & (1u << i)) {
                        size_t relativeIdx1 = MarchingCube::s_CornerIndices[i * 2];
                        size_t relativeIdx2 = MarchingCube::s_CornerIndices[i * 2 + 1];
                        size_t absoluteIdx1 = gridCornerIndices[relativeIdx1] - 1;
                        size_t absoluteIdx2 = gridCornerIndices[relativeIdx2] - 1;
                        if (absoluteIdx2 < absoluteIdx1) {
                            // We want unique identifier of an edge
                            std::swap(absoluteIdx1, absoluteIdx2);
                        }

                        bool found = false;
#pragma omp critical(mapAccess)
                        {
                            auto it = m_VertexIndices.find({absoluteIdx1, absoluteIdx2});
                            if (it != m_VertexIndices.end()) {
                                // Interpolated edge vertex was already computed, retrieve its index
                                found = true;
                                intersections[i] = it->second;
                            }
                        }

                        if (!found) {
                            // Edge vertex doesn't exist yet. Compute it and store index
                            const glm::vec3 &p1(cubeCorners[relativeIdx1]);
                            const glm::vec3 &p2(cubeCorners[relativeIdx2]);
                            float l0 = isoValues[relativeIdx1];
                            float l1 = isoValues[relativeIdx2];
                            const float interpCoeff = (0 - l0) / (l1 - l0);

                            // New index
#pragma omp critical(vertexEmit)
                            {
                                intersections[i] = m_MeshVertices.size();
                                // New vertex with the new index position
                                m_MeshVertices.emplace_back(
                                        p1.x * (1.0f - interpCoeff) + p2.x * interpCoeff,
                                        p1.y * (1.0f - interpCoeff) + p2.y * interpCoeff,
                                        p1.z * (1.0f - interpCoeff) + p2.z * interpCoeff
                                );
                            }

#pragma omp critical(mapAccess)
                            m_VertexIndices[{absoluteIdx1, absoluteIdx2}] = intersections[i];
                        }
                    }
                }

                // Assemble triangles
                const int8_t *configTriangles = &triangleTable[cubeIdx][0];
                std::array<GLuint, 3> triangleIndices{};
                for (size_t i = 0; configTriangles[i] != -1; i += 3) {
                    triangleIndices = {
                            intersections[configTriangles[i]],
                            intersections[configTriangles[i + 1]],
                            intersections[configTriangles[i + 2]]
                    };

#pragma omp critical(indexEmit)
                    m_MeshIndices.insert(m_MeshIndices.end(), triangleIndices.begin(), triangleIndices.end());
                }
            }
        }
    }

    BufferData();
}

void CloudModel::MLSReconstruction() {
    ClockGuard timer(__func__);

    m_Grid.Regenerate();


    m_Grid.CalculateIsoValuesMLS(m_NeighbourhoodSize);

    m_VertexIndices.clear();
    m_VertexIndices.rehash(m_Grid.m_Points.size() * 1.25f);
    m_MeshVertices.clear();
    m_MeshIndices.clear();

    size_t zPointCount = m_Grid.GetResZ() + 1;
    size_t yzPointCount = (m_Grid.GetResY() + 1) * zPointCount;

    std::array<GLuint, 12> intersections{};
    std::array<glm::vec3, 8> cubeCorners{};
    std::array<float, 8> isoValues{};

#pragma omp parallel for schedule(static) firstprivate(intersections, cubeCorners, isoValues, zPointCount, yzPointCount) shared(edgeTable, triangleTable)
    for (size_t x = 0; x < m_Grid.GetResX(); ++x) {
        for (size_t y = 0; y < m_Grid.GetResY(); ++y) {
            for (size_t z = 0; z < m_Grid.GetResZ(); ++z) {
                std::array<size_t, 8> gridCornerIndices{
                        z + (y * zPointCount) + (x * yzPointCount),
                        1 + z + (y * zPointCount) + (x * yzPointCount),
                        1 + z + ((y + 1) * zPointCount) + (x * yzPointCount),
                        z + ((y + 1) * zPointCount) + (x * yzPointCount),

                        z + (y * zPointCount) + ((x + 1) * yzPointCount),
                        1 + z + (y * zPointCount) + ((x + 1) * yzPointCount),
                        1 + z + ((y + 1) * zPointCount) + ((x + 1) * yzPointCount),
                        z + ((y + 1) * zPointCount) + ((x + 1) * yzPointCount)
                };

                for (size_t i = 0; i < cubeCorners.size(); ++i) {
                    cubeCorners[i] = m_Grid.m_Points[gridCornerIndices[i]].pos;
                    isoValues[i] = m_Grid.m_IsoValues[gridCornerIndices[i]];
                }

                uint32_t cubeIdx = 0;
                for (size_t i = 0; i < cubeCorners.size(); ++i)
                    cubeIdx |= unsigned(isoValues[i] < 0) << i;

                /* Cube is entirely in/out of the surface */
                if (cubeIdx == 0 || cubeIdx == 255)
                    continue;

                // Find the vertices where the surface intersects the cube
                uint32_t cubeConfig = edgeTable[cubeIdx];
                for (size_t i = 0; i < intersections.size(); ++i) {
                    if (cubeConfig & (1u << i)) {
                        size_t relativeIdx1 = MarchingCube::s_CornerIndices[i * 2];
                        size_t relativeIdx2 = MarchingCube::s_CornerIndices[i * 2 + 1];
                        size_t absoluteIdx1 = gridCornerIndices[relativeIdx1] - 1;
                        size_t absoluteIdx2 = gridCornerIndices[relativeIdx2] - 1;
                        if (absoluteIdx2 < absoluteIdx1) {
                            // We want unique identifier of an edge
                            std::swap(absoluteIdx1, absoluteIdx2);
                        }

                        bool found = false;
#pragma omp critical(mapAccess)
                        {
                            auto it = m_VertexIndices.find({absoluteIdx1, absoluteIdx2});
                            if (it != m_VertexIndices.end()) {
                                // Interpolated edge vertex was already computed, retrieve its index
                                found = true;
                                intersections[i] = it->second;
                            }
                        }

                        if (!found) {
                            // Edge vertex doesn't exist yet. Compute it and store index
                            const glm::vec3 &p1(cubeCorners[relativeIdx1]);
                            const glm::vec3 &p2(cubeCorners[relativeIdx2]);
                            float l0 = isoValues[relativeIdx1];
                            float l1 = isoValues[relativeIdx2];
                            const float interpCoeff = (0 - l0) / (l1 - l0);

                            // New index
#pragma omp critical(vertexEmit)
                            {
                                intersections[i] = m_MeshVertices.size();
                                // New vertex with the new index position
                                m_MeshVertices.emplace_back(
                                        p1.x * (1.0f - interpCoeff) + p2.x * interpCoeff,
                                        p1.y * (1.0f - interpCoeff) + p2.y * interpCoeff,
                                        p1.z * (1.0f - interpCoeff) + p2.z * interpCoeff
                                );
                            }

#pragma omp critical(mapAccess)
                            m_VertexIndices[{absoluteIdx1, absoluteIdx2}] = intersections[i];
                        }
                    }
                }

                // Assemble triangles
                const int8_t *configTriangles = &triangleTable[cubeIdx][0];
                std::array<GLuint, 3> triangleIndices{};
                for (size_t i = 0; configTriangles[i] != -1; i += 3) {
                    triangleIndices = {
                            intersections[configTriangles[i]],
                            intersections[configTriangles[i + 1]],
                            intersections[configTriangles[i + 2]]
                    };

#pragma omp critical(indexEmit)
                    m_MeshIndices.insert(m_MeshIndices.end(), triangleIndices.begin(), triangleIndices.end());
                }
            }
        }
    }

    BufferData();
}

void Grid::CalculateIsoValuesMLS(size_t neighbourhoodSize) {
    pcl::IndicesPtr indices(new std::vector<int>);
    indices->resize(neighbourhoodSize);
    pcl::IndicesPtr _closest_ind(new std::vector<int>);
    _closest_ind->resize(1);

    std::vector<float> k_sqr_distances;
    std::vector<float> _closest_k_sqr_distances;
    unsigned degree = neighbourhoodSize > 9 ? 3 : neighbourhoodSize > 5 ? 2 : 1;
    size_t mat_size = (degree == 3) ? 9 : ((degree == 2) ? 6 : 3);
    ClockGuard timer(__func__);

    size_t startIdx = 0;
    size_t endIdx = m_Points.size();

    for (size_t i = startIdx; i < endIdx; ++i) {
        float distance = 0.0f;
        VertexRGB &gridVertex(m_Points[i]);
        pcl::PointNormal gridPoint;
        std::memcpy(gridPoint.data, glm::value_ptr(gridVertex.pos), sizeof(float) * 3);

        m_ParentModel.m_Tree->nearestKSearch(gridPoint, 1, *_closest_ind, _closest_k_sqr_distances);
        pcl::PointNormal closest_point = m_ParentModel.m_Cloud->at(_closest_ind->at(0));
        int found = m_ParentModel.m_Tree->nearestKSearch(gridPoint, neighbourhoodSize, *indices, k_sqr_distances);
        assert((size_t) found == neighbourhoodSize);
        Eigen::VectorXd x_s(found);
        Eigen::VectorXd y_s(found);
        Eigen::VectorXd z_s(found);
        float sums2[3] = {0,0,0};
        float sums[3] = {0,0,0};
        for (size_t _i = 0; _i < indices->size(); _i++) {
            float _x = m_ParentModel.m_Cloud->at(indices->at(_i)).x;
            x_s[_i] = _x;
            sums2[0] += pow(_x,2);
            sums[0] += _x;

            float _y = m_ParentModel.m_Cloud->at(indices->at(_i)).y;
            y_s[_i] = _y;
            sums2[1] += pow(_y,2);
            sums[1] += _y;

            float _z = m_ParentModel.m_Cloud->at(indices->at(_i)).z;
            z_s[_i] = _z;
            sums2[2] += pow(_z,2);
            sums[2] += _z;
        }

        double vars[3] = {sums2[0]/found - pow(sums[0]/found, 2),sums2[1]/found - pow(sums[1]/found, 2),sums2[2]/found - pow(sums[2]/found, 2)};
        /* polynom order + 1 -> second argument of mat constructor */
        auto pseudoinverse = [](Eigen::MatrixXd mat) { return (mat.transpose() * mat).inverse() * mat.transpose(); };
        auto dist2d = [](float p, float inter, float dist) { return (p - inter) * exp(-dist); };
        void (*mat2d)(Eigen::MatrixXd &mat, int r, const Eigen::VectorXd &x_1, const Eigen::VectorXd &x_2);
        float (*_f_2d)(Eigen::MatrixXd coeffs, float x_1, float x_2);
        if (degree == 3) {
            mat2d = &mat2d_3;
            _f_2d = &_f_2d_3;
        } else if (degree == 2) {
            mat2d = &mat2d_2;
            _f_2d = &_f_2d_2;
        }
        else if (degree == 1) {
            mat2d = &mat2d_1;
            _f_2d = &_f_2d_1;
        } else
            std::cerr << "unsupported degree of polynom: " << degree << std::endl;
        /* now decide which coordinate will be result and which base for function approximation */
        if (vars[0] < vars[1] && vars[0] < vars[2]) {
            /* x is y */
            /* approximate f(y, z) = x */
            Eigen::MatrixXd mat_yz(found, mat_size);
            for (int _i = 0; _i < found; _i++){
                mat2d(mat_yz, _i, y_s, z_s);
            }
            /* fit data */
            Eigen::MatrixXd coeffs_yz = pseudoinverse(mat_yz) * x_s;
            /* compute distance */
            distance = (dist2d(closest_point.x - closest_point.normal_x, _f_2d(coeffs_yz, closest_point.y, closest_point.z), 1) < 0 ? 1 : -1 ) *
                           dist2d(gridPoint.x, _f_2d(coeffs_yz, gridPoint.y, gridPoint.z), _closest_k_sqr_distances.at(0));

        } else if (vars[1] < vars[0] && vars[1] < vars[2]) {
            /* y is y */
            /* approximate f(x, z) = y */
            Eigen::MatrixXd mat_xz(found, mat_size);
            for (int _i = 0; _i < found; _i++){
                mat2d(mat_xz, _i, x_s, z_s);
            }
            Eigen::MatrixXd coeffs_xz = pseudoinverse(mat_xz) * y_s;
            distance = (dist2d(closest_point.y - closest_point.normal_y, _f_2d(coeffs_xz, closest_point.x, closest_point.z), 1) < 0 ? 1 : -1 ) *
                        dist2d(gridPoint.y, _f_2d(coeffs_xz, gridPoint.x, gridPoint.z), _closest_k_sqr_distances.at(0));
        } else {
            /* z is y */
            /* approximate f(x, y) = z */
            Eigen::MatrixXd mat_xy(found, mat_size);
            for (int _i = 0; _i < found; _i++){
                mat2d(mat_xy, _i, x_s, y_s);
            }
            Eigen::MatrixXd coeffs_xy = pseudoinverse(mat_xy) * z_s;
            distance = (dist2d(closest_point.z - closest_point.normal_z, _f_2d(coeffs_xy, closest_point.x, closest_point.y), 1) < 0 ? 1 : -1 ) *
                        dist2d(gridPoint.z, _f_2d(coeffs_xy, gridPoint.x, gridPoint.y), _closest_k_sqr_distances.at(0));
        }
        // smallest variance is variable



        if (neighbourhoodSize > 1) {/*
            for (size_t vectorIdx = 0; vectorIdx < 3; vectorIdx++) {
                auto eigenVector(eigenVectors.col(vectorIdx));
                eigenVector.normalize();

                float dotProduct = nearestPoint.getNormalVector3fMap().dot(eigenVector);
                if (std::abs(dotProduct) >= max) {
                    max = dotProduct;

                    // Resulting normal (eigen vector) might be incorrectly oriented. Original
                    // paper deals with this by using complex algorithm which traverses Riemannian
                    // graph and consistently orients normals of connected tangent planes.
                    // We'll use a hack and orient the normal based on the orientation of the closest point
                    normal = glm::vec3(eigenVector.x(), eigenVector.y(), eigenVector.z());
                    if (max < 0) {
                        max = -max;
                        normal = -normal;
                    }
                }
            }

            glm::vec3 direction(gridVertex.pos - centroidVertex);
            distance = glm::dot(direction, glm::normalize(normal));*/
        } else {
            pcl::PointNormal nearestPoint = m_ParentModel.m_Cloud->points[indices->front()];

            // Calculate the distance between the MC corner point and the tangent
            // plane of the closest surface point. Dot projection of the point2point
            // vector and the unit-length surface normal gives us the distance.
            auto direction = gridPoint.getVector3fMap() - nearestPoint.getVector3fMap();
            distance = nearestPoint.getNormalVector3fMap().dot(direction);
        }

        m_IsoValues[i] = distance;

        // Red color for corner points outside the geometry and green for points that are inside
        gridVertex.color = distance <= 0 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    }

    // Update grid vertex data with colors
    glNamedBufferSubData(m_VBO, 0, sizeof(VertexRGB) * m_Points.size(), m_Points.data());
}


void CloudModel::BufferData() {
    ClockGuard timer(__func__);

    // Buffer model vertex and index data and bind buffers to VAO
    glNamedBufferData(m_VBOs[Mesh], sizeof(glm::vec3) * m_MeshVertices.size(), m_MeshVertices.data(), GL_STATIC_DRAW);
    glNamedBufferData(m_EBOs[Mesh], sizeof(GLuint) * m_MeshIndices.size(), m_MeshIndices.data(), GL_STATIC_DRAW);
    glVertexArrayElementBuffer(m_VAOs[Mesh], m_EBOs[Mesh]);
    glEnableVertexArrayAttrib(m_VAOs[Mesh], 0);
    glVertexArrayAttribFormat(m_VAOs[Mesh], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[Mesh], 0, m_VBOs[Mesh], 0, sizeof(glm::vec3));
}


void CloudModel::ExtractPclReconstructionData(const std::vector<pcl::Vertices> &outputIndices,
                                              const pcl::PointCloud<pcl::PointNormal>::Ptr &surfacePoints) {
    // Copy surface vertices
    m_MeshVertices.clear();
    m_MeshVertices.resize(surfacePoints->size());
    for (size_t i = 0; i < surfacePoints->size(); ++i) {
        memcpy(glm::value_ptr(m_MeshVertices[i]), surfacePoints->points[i].data, sizeof(float) * 3);
    }

    // PCL uses weird memory layout for indices, copy the data
    // into new vector with flat layout so that it can be used by OpenGL
    size_t idx = 0;
    m_MeshIndices.clear();
    m_MeshIndices.resize(outputIndices.size() * 3);
    for (size_t i = 0; i < outputIndices.size(); i++) {
        const auto &indices = outputIndices[i];
        for (size_t index : indices.vertices)
            m_MeshIndices[idx++] = index;
    }
}


void CloudModel::PCL_HoppeReconstruction() {
    m_Grid.Regenerate();

    ClockGuard timer(__func__);

    // Reconstruction
    pcl::MarchingCubesHoppe<pcl::PointNormal> hoppe;
    hoppe.setInputCloud(m_Cloud);
    hoppe.setPercentageExtendGrid(0.1f);
    hoppe.setGridResolution(m_Grid.GetResX(), m_Grid.GetResY(), m_Grid.GetResZ());
    hoppe.setSearchMethod(m_Tree);
    hoppe.setIsoLevel(m_IsoLevel);
    hoppe.setDistanceIgnore(m_IgnoreDistance);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    hoppe.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();

    // TODO: Move to CloudModel later on
    // std::vector<GLuint> flatIndices;
    // pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    // auto &surfacePoints = surfaceCloud->points;
    // std::cout << "Cloud size: " << surfacePoints.size() << std::endl;
    // std::cout << "Triangle count: " << (flatIndices.size() / 3) << std::endl;

    // Simple caching system for later (we might want to display multiple models during presentation?)
//    std::ifstream indexCache("index_cache.dat", std::ios::in | std::ios::binary);
//    if (false && pcl::io::loadPCDFile<pcl::PointNormal>("surface_cache.pcd", *surfaceCloud) >= 0 &&
//        indexCache.is_open()) {
//        // Cache exists
//        size_t cacheSize;
//        indexCache.read((char *) &cacheSize, sizeof(size_t));
//        flatIndices.resize(cacheSize);
//        indexCache.read((char *) flatIndices.data(), cacheSize * sizeof(GLuint));
//    } else {
//        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals;
//        // if (false)
//        //     cloud_point_normals = bunnyWithOutNormals();
//        // else
//        //     cloud_point_normals = bunnyWithNormals();
//
//
//        // Save cache
//        pcl::io::savePCDFileASCII("surface_cache.pcd", *surfaceCloud);
//        std::ofstream cacheFile("index_cache.dat", std::ios::out | std::ios::binary);
//        size_t cacheSize = flatIndices.size();
//        cacheFile.write((char *) &cacheSize, sizeof(size_t));
//        cacheFile.write((char *) flatIndices.data(), cacheSize * sizeof(GLuint));
//        cacheFile.close();
//    }
}


void CloudModel::PCL_PoissonReconstruction() {
    ClockGuard timer(__func__);

    // Reconstruction
    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setInputCloud(m_Cloud);
    poisson.setSearchMethod(m_Tree);
    poisson.setDegree(m_Degree);
    poisson.setDepth(m_Depth);
    poisson.setMinDepth(m_MinDepth);
    poisson.setIsoDivide(m_IsoDivide);
    poisson.setSolverDivide(m_SolverDivide);
    poisson.setPointWeight(m_PointWeight);
    poisson.setSamplesPerNode(m_SamplesPerNode);
    poisson.setScale(m_Scale);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    poisson.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();
}


ProgramObject CloudModel::s_MeshProgram;
ProgramObject CloudModel::s_ShaderProgram;
ProgramObject CloudModel::s_GeometryProgram;
ProgramObject CloudModel::s_ColorProgram;