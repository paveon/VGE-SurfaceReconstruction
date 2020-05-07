#include "CloudModel.h"
#include "MCTable.h"

#include <chrono>
#include <unordered_set>
#include <queue>
#include <stack>

#include <pcl/common/pca.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/geometry.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/impl/organized_fast_mesh.hpp>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <Eigen/Dense>
#include "MLS_helper_functions.cpp"
#include "DisjointSet.h"

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
        std::cout << "[Timer] " << m_Name << ": " << ElapsedTime() << "s" << std::endl;
    }

    double ElapsedTime() const {
        std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - m_Start;
        return elapsed_seconds.count();
    }
};

void BoundingBox::Draw(ProgramObject &shader, glm::mat4 pvm) const {
    // Draw grid points
    glm::vec3 color(1.0f, 0.0f, 0.0f);
    shader.use();
    shader.setMatrix4fv("pvm", glm::value_ptr(pvm));
    shader.set3fv("primitiveColor", glm::value_ptr(color));
    glBindVertexArray(m_VAO);
    glDrawElements(GL_LINES, Cube::s_CornerIndices.size(), GL_UNSIGNED_INT, nullptr);
}


void Grid::Regenerate() {
    if (!m_Invalidated)
        return;

    ClockGuard timer(__func__);

    BoundingBox bb(m_ParentModel.m_BB);
    float deltaX = bb.size.x / (float) m_ResX;
    float deltaY = bb.size.y / (float) m_ResY;
    float deltaZ = bb.size.z / (float) m_ResZ;
    size_t yzPointCount = (m_ResZ + 1) * (m_ResY + 1);
    size_t pointsTotal = (m_ResX + 1) * yzPointCount;
    bool glBufferOverflow = pointsTotal > m_Points.size();

    m_Points.resize(pointsTotal);
    m_IsoValues.resize(pointsTotal);

#pragma omp parallel for schedule(static) firstprivate(deltaX, deltaY, deltaZ, bb, yzPointCount)
    for (size_t x = 0; x < m_ResX + 1; ++x) {
        size_t gridIdx = x * yzPointCount;
        float xPos = bb.min.x + (x * deltaX);

        for (size_t y = 0; y < m_ResY + 1; ++y) {
            float yPos = bb.min.y + (y * deltaY);

            for (size_t z = 0; z < m_ResZ + 1; ++z) {
                float zPos = bb.min.z + (z * deltaZ);
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
            if (neighbourhoodSize > 2) {
                pcl::PCA<pcl::PointNormal> pca;
                pca.setInputCloud(m_ParentModel.m_Cloud);
                pca.setIndices(indices);
                auto centroid = pca.getMean();
                auto eigenVectors = pca.getEigenVectors();
                glm::vec3 centroidVertex(centroid.x(), centroid.y(), centroid.z());
                pcl::PointNormal nearestPoint = m_ParentModel.m_Cloud->points[indices->front()];
                Eigen::Vector3f averageNormal = nearestPoint.getNormalVector3fMap();
                for (size_t neighbourIdx = 1; i < neighbourhoodSize; i++) {
                    averageNormal += m_ParentModel.m_Cloud->points[indices->at(neighbourIdx)].getNormalVector3fMap();
                }
                averageNormal /= neighbourhoodSize;

                // Find eigen vector with lowest angle
                float max = 0;
                glm::vec3 normal;
                for (size_t vectorIdx = 0; vectorIdx < 3; vectorIdx++) {
                    auto eigenVector(eigenVectors.col(vectorIdx));
                    eigenVector.normalize();

//                    float dotProduct = nearestPoint.getNormalVector3fMap().dot(eigenVector);
                    float dotProduct = averageNormal.dot(eigenVector);
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


CloudModel::CloudModel(const std::string& name, pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir) :
        m_Name(name),
        m_Cloud(std::move(cloud)),
        m_Tree(new pcl::search::KdTree<pcl::PointNormal>()),
        m_BB(m_Cloud, 0.1f),
        m_Grid(*this) {

    m_Tree->setInputCloud(m_Cloud);
    std::cout << "[Cloud] Organized: " << m_Cloud->isOrganized() << std::endl;

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
        s_MeshShader = createProgram(meshVS, meshFS);
    }

    glCreateVertexArrays(m_VAOs.size(), m_VAOs.data());
    glCreateBuffers(m_VBOs.size(), m_VBOs.data());
    glCreateBuffers(m_EBOs.size(), m_EBOs.data());

    size_t pointCount = m_Cloud->points.size();

    // Buffer point cloud data and setup vertex attributes
    m_CloudNormals.resize(pointCount * 2);
    for (size_t i = 0; i < pointCount; i += 2) {
        const auto &cloudPt = m_Cloud->points[i];
        memcpy(glm::value_ptr(m_CloudNormals[i]), cloudPt.data, sizeof(float) * 3);
        auto endPoint = cloudPt.getVector3fMap() + (cloudPt.getNormalVector3fMap() * 0.2f);
        m_CloudNormals[i + 1] = glm::vec3(endPoint.x(), endPoint.y(), endPoint.z());
    }
    glNamedBufferData(m_VBOs[CloudNormals], sizeof(glm::vec3) * m_CloudNormals.size(), m_CloudNormals.data(), GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[CloudNormals], 0);
    glVertexArrayAttribFormat(m_VAOs[CloudNormals], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[CloudNormals], 0, m_VBOs[CloudNormals], 0, sizeof(glm::vec3));

    glNamedBufferData(m_VBOs[Cloud], sizeof(pcl::PointNormal) * pointCount, m_Cloud->points.data(), GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[Cloud], 0);
    glEnableVertexArrayAttrib(m_VAOs[Cloud], 1);
    glVertexArrayAttribFormat(m_VAOs[Cloud], 0, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, data));
    glVertexArrayAttribFormat(m_VAOs[Cloud], 1, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, normal));
    glVertexArrayVertexBuffer(m_VAOs[Cloud], 0, m_VBOs[Cloud], 0, sizeof(pcl::PointNormal));
    glVertexArrayVertexBuffer(m_VAOs[Cloud], 1, m_VBOs[Cloud], 0, sizeof(pcl::PointNormal));

    // Setup VBO for spanning tree edges
    glNamedBufferData(m_VBOs[SpanningTree], sizeof(glm::vec3) * (pointCount - 1) * 2, nullptr, GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[SpanningTree], 0);
    glVertexArrayAttribFormat(m_VAOs[SpanningTree], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[SpanningTree], 0, m_VBOs[SpanningTree], 0, sizeof(glm::vec3));

    m_Grid.Regenerate();
}


struct GraphEdge {
    size_t m_Src = 0;
    size_t m_Dst = 0;
    float m_Weight = 0.0f;

    GraphEdge() = default;
};


void CloudModel::EstimateNormals(){
    if (m_NormalsEstimated)
        return;

    size_t pointCount = m_Cloud->points.size();
    std::vector<int> pointIndices(pointCount);
    std::iota (pointIndices.begin(), pointIndices.end(), 0);
    std::vector<std::vector<int>> allKNNs(pointCount);
    std::vector<std::vector<float>> allDistances(pointCount);
    size_t neighbourCount = 32;
    for (size_t i = 0; i < pointCount; i++) {
        allKNNs[i].resize(neighbourCount);
        allDistances[i].resize(neighbourCount);
    }

    // Use less KNNs for normal estimation
    size_t K = 8;
    std::vector<int> estimationKNNs(K);

    m_Tree->nearestKSearch(*m_Cloud, pointIndices, neighbourCount, allKNNs, allDistances);

    pcl::NormalEstimation<pcl::PointNormal, pcl::Normal> ne;
    std::unordered_set<size_t> visitedNodes;
    std::vector<GraphEdge> graphEdges;
    graphEdges.resize((pointCount * neighbourCount) - pointCount);
    visitedNodes.rehash(pointCount * 1.25f);

    // Create graph edges and calculate distances
    size_t edgeIdx = 0;
    for (size_t srcIdx = 0; srcIdx < pointCount; srcIdx++) {
        pcl::PointNormal& srcNode = m_Cloud->points[srcIdx];
        const std::vector<int>& srcNeighbours = allKNNs[srcIdx];

        // Estimate normal for all points
        memcpy(estimationKNNs.data(), srcNeighbours.data(), sizeof(int) * estimationKNNs.size());
        ne.computePointNormal(*m_Cloud, estimationKNNs, srcNode.normal_x, srcNode.normal_y, srcNode.normal_z, srcNode.curvature);

        for (size_t i = 1; i < srcNeighbours.size(); i++) {
            size_t dstIdx = srcNeighbours[i];
            // Create edge for each pair of nodes
            pcl::PointNormal& dstNode = m_Cloud->points[dstIdx];
            float distance = pcl::geometry::squaredDistance(srcNode, dstNode);
            graphEdges[edgeIdx++] = GraphEdge{srcIdx, dstIdx, distance};
        }
    }

    std::sort(graphEdges.begin(), graphEdges.end(), [&](const GraphEdge& e1, const GraphEdge& e2) {
        return e1.m_Weight < e2.m_Weight;
    });

    DisjointSetForest<int> emstForest(pointIndices);
    std::vector<GraphEdge> emstEdges;
    size_t emstEdgeCount = pointCount - 1;
    emstEdges.resize(emstEdgeCount);

    // Filter all edges and create Euclidean Minimal Spanning Tree (EMST)
    size_t edgeCount = 0;
    for (const GraphEdge& edge : graphEdges) {
        size_t srcSet = emstForest.find_set(edge.m_Src);
        size_t dstSet = emstForest.find_set(edge.m_Dst);
        if (srcSet != dstSet)
        {
            emstEdges[edgeCount++] = edge;

            if (edgeCount == emstEdgeCount)
                break;

            emstForest.union_sets(srcSet, dstSet);
        }
    }

    if (edgeCount < emstEdgeCount) {
        emstEdgeCount = edgeCount;
        emstEdges.resize(emstEdgeCount);
    }
    emstEdges.reserve(emstEdgeCount + (emstEdgeCount * (K / 2)));

    // Enrich the EMST with neighbouring edges and replace distance
    // based edge weight with normal angle difference weight
    for (edgeIdx = 0; edgeIdx < emstEdgeCount; edgeIdx++) {
        size_t edgeSrc = emstEdges[edgeIdx].m_Src;
        size_t edgeDst = emstEdges[edgeIdx].m_Dst;
        const pcl::PointNormal& srcNode = m_Cloud->points[edgeSrc];
        const pcl::PointNormal& dstNode = m_Cloud->points[edgeDst];
        emstEdges[edgeIdx].m_Weight = std::abs(srcNode.getNormalVector3fMap().dot(dstNode.getNormalVector3fMap()));


        memcpy(estimationKNNs.data(), allKNNs[edgeSrc].data(), sizeof(int) * estimationKNNs.size());
        for (size_t i = 1; i < estimationKNNs.size(); i++) {
            size_t dstIdx = estimationKNNs[i];

            pcl::PointNormal& srcNeighbour = m_Cloud->points[dstIdx];
            float weight = std::abs(srcNode.getNormalVector3fMap().dot(srcNeighbour.getNormalVector3fMap()));
            emstEdges.emplace_back(GraphEdge{edgeSrc, dstIdx, weight});
        }

        memcpy(estimationKNNs.data(), allKNNs[edgeDst].data(), sizeof(int) * estimationKNNs.size());
        for (size_t i = 1; i < estimationKNNs.size(); i++) {
            size_t dstIdx = estimationKNNs[i];

            pcl::PointNormal& dstNeighbour = m_Cloud->points[dstIdx];
            float weight = std::abs(dstNode.getNormalVector3fMap().dot(dstNeighbour.getNormalVector3fMap()));
            emstEdges.emplace_back(GraphEdge{edgeDst, dstIdx, weight});
        }
    }

    // Sort edges by weight (which is different than before)
    std::sort(emstEdges.begin(), emstEdges.end(), [&](const GraphEdge& e1, const GraphEdge& e2) {
        return e1.m_Weight > e2.m_Weight;
    });

    // Create disjoint sets
    DisjointSetForest<int> forest(pointIndices);
    std::vector<std::vector<size_t>> spanningTree(pointCount);
    m_TreeEdges.resize((emstEdgeCount) * 2);

    // Iterate through all enriched EMST edges and create Maximal Spanning Tree
    edgeCount = 0;
    std::unordered_set<std::pair<size_t, size_t>, PairHash> existingEdges;
    existingEdges.rehash(emstEdgeCount * 1.25);
    for (const GraphEdge& edge : emstEdges) {
        auto key = edge.m_Src < edge.m_Dst ? std::make_pair(edge.m_Src, edge.m_Dst) : std::make_pair(edge.m_Dst, edge.m_Src);
        if (existingEdges.find(key) != existingEdges.end())
            continue;

        existingEdges.insert(key);

        size_t srcSet = forest.find_set(edge.m_Src);
        size_t dstSet = forest.find_set(edge.m_Dst);
        if (srcSet != dstSet)
        {
            spanningTree[edge.m_Src].push_back(edge.m_Dst);
            spanningTree[edge.m_Dst].push_back(edge.m_Src);
            const pcl::PointNormal& p1 = m_Cloud->points[edge.m_Src];
            const pcl::PointNormal& p2 = m_Cloud->points[edge.m_Dst];
            memcpy(glm::value_ptr(m_TreeEdges[edgeCount * 2]), p1.data, sizeof(float) * 3);
            memcpy(glm::value_ptr(m_TreeEdges[(edgeCount * 2) + 1]), p2.data, sizeof(float) * 3);
            edgeCount++;

            if (edgeCount == emstEdgeCount)
                break;

            // Merge two sets
            forest.union_sets(srcSet, dstSet);
        }
    }

    if (edgeCount < emstEdgeCount) {
        m_TreeEdges.resize(edgeCount * 2);
    }

    // Find point with highest Z value and orient its normal upwards
    float maxZ = m_Cloud->points[0].z;
    float maxIdx = 0;
    for (size_t i = 1; i < m_Cloud->points.size(); ++i) {
        if (m_Cloud->points[i].z > maxZ) {
            maxIdx = i;
            maxZ = m_Cloud->points[i].z;
        }
    }

    pcl::PointNormal& maxPoint = m_Cloud->points[maxIdx];
    Eigen::Vector3f upVector(0.0f, 0.0f, 1.0f);
    if (maxPoint.getNormalVector3fMap().dot(upVector) < 0) {
        // Normal of point with maximum z-coord should point upwards, invert normal
        maxPoint.normal_x *= -1;
        maxPoint.normal_y *= -1;
        maxPoint.normal_z *= -1;
    }

    // Propagate the orientation through the MST from the max point
    std::stack<std::pair<size_t, size_t>> toVisit;
    toVisit.push(std::make_pair(maxIdx, maxIdx));
    while (!toVisit.empty()) {
        auto[parentIdx, currentNodeIdx] = toVisit.top();
        toVisit.pop();
        const pcl::PointNormal& currentNode = m_Cloud->points[currentNodeIdx];

        const std::vector<size_t>& childrenIndices(spanningTree[currentNodeIdx]);
        for (size_t childIdx : childrenIndices) {
            if (childIdx == parentIdx)
                continue;

            toVisit.push(std::make_pair(currentNodeIdx, childIdx));
            pcl::PointNormal& childNode = m_Cloud->points[childIdx];
            // Flip normal orientation if necessary
            if (currentNode.getNormalVector3fMap().dot(childNode.getNormalVector3fMap()) < 0) {
                // Normal of point with maximum z-coord should point upwards, invert normal
                childNode.normal_x *= -1;
                childNode.normal_y *= -1;
                childNode.normal_z *= -1;
            }
        }
    }

    glNamedBufferSubData(m_VBOs[SpanningTree], 0, sizeof(glm::vec3) * edgeCount * 2, m_TreeEdges.data());
    glNamedBufferSubData(m_VBOs[Cloud], 0, sizeof(pcl::PointNormal) * m_Cloud->points.size(), m_Cloud->points.data());
    m_NormalsEstimated = true;
}


void CloudModel::Draw(glm::mat4 pv, glm::vec3 color) const {
    glm::mat4 modelMatrix(1.0f);
    glm::mat4 pvm = pv * modelMatrix;
    glm::vec3 outlineColor(1.0f, 0.0f, 0.0f);

    if (m_ShowMesh && !m_MeshVertices.empty()) {
        // Draw model
        s_MeshShader.use();
        s_MeshShader.set3fv("primitiveColor", glm::value_ptr(color));
        s_MeshShader.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(m_VAOs[Mesh]);
        glDrawElements(GL_TRIANGLES, m_MeshIndices.size(), GL_UNSIGNED_INT, nullptr);

        // Draw outlines of model triangles, it looks weird without illumination model otherwise
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        s_MeshShader.set3fv("primitiveColor", glm::value_ptr(outlineColor));
        glDrawElements(GL_TRIANGLES, m_MeshIndices.size(), GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    if (m_ShowSpanningTree) {
        glm::vec3 treeColor(1.0f, 0.0f, 1.0f);
        s_MeshShader.use();
        s_MeshShader.set3fv("primitiveColor", glm::value_ptr(treeColor));
        s_MeshShader.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glBindVertexArray(m_VAOs[SpanningTree]);
        glDrawArrays(GL_LINES, 0, m_TreeEdges.size());
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
    }

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

    if (m_ShowGrid)
        m_Grid.Draw(s_ColorProgram, pvm);

    if (m_ShowBB)
        m_BB.Draw(s_MeshShader, pvm);
}


double CloudModel::Reconstruct(ReconstructionMethod method) {
    EstimateNormals();
    switch (method) {
        case ReconstructionMethod::ModifiedHoppe:
            return HoppeReconstruction();

        case ReconstructionMethod::MLS:
            return MLSReconstruction();

        case ReconstructionMethod::PCL_Hoppe:
            return PCL_HoppeReconstruction();

        case ReconstructionMethod::PCL_MarchingCubesRBF:
            return PCL_MC_RBF_Reconstruction();

        case ReconstructionMethod::PCL_Poisson:
            return PCL_PoissonReconstruction();

        case ReconstructionMethod::PCL_ConcaveHull:
            return PCL_ConcaveHullReconstruction();

        case ReconstructionMethod::PCL_ConvexHull:
            return PCL_ConvexHullReconstruction();

        case ReconstructionMethod::PCL_GreedyProjectionTriangulation:
            return PCL_GP3();

        case ReconstructionMethod::PCL_OrganizedFastMesh:
            return PCL_OrganizedFastMeshReconstruction();
    }

    return 0;
}


void CloudModel::MarchCubes() {
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
                        size_t relativeIdx1 = Cube::s_CornerIndices[i * 2];
                        size_t relativeIdx2 = Cube::s_CornerIndices[i * 2 + 1];
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
}


double CloudModel::HoppeReconstruction() {
    ClockGuard timer(__func__);

    m_Grid.Regenerate();
    m_Grid.CalculateIsoValues(m_NeighbourhoodSize);

    MarchCubes();

    BufferData();

    return timer.ElapsedTime();
}

double CloudModel::MLSReconstruction() {
    ClockGuard timer(__func__);

    m_Grid.Regenerate();
    m_Grid.CalculateIsoValuesMLS(m_NeighbourhoodSize);
    if(m_MLS_use_median) {
        float kernel[3];
        for (unsigned i = 1; i < m_Grid.m_IsoValues.size() - 1; i++) {
            kernel[0] = m_Grid.m_IsoValues[i - 1];
            kernel[1] = m_Grid.m_IsoValues[i];
            kernel[2] = m_Grid.m_IsoValues[i + 1];
            std::sort(kernel, kernel + 3, std::greater<float>());
            m_Grid.m_IsoValues[i] = kernel[1];
        }
    }
    MarchCubes();

    BufferData();

    return timer.ElapsedTime();
}

void Grid::CalculateIsoValuesMLS(size_t neighbourhoodSize) {
    pcl::IndicesPtr indices(new std::vector<int>);
    indices->resize(neighbourhoodSize);
    pcl::IndicesPtr _closest_ind(new std::vector<int>);
    _closest_ind->resize(1);

    std::vector<float> k_sqr_distances;
    std::vector<float> _closest_k_sqr_distances;
    unsigned degree;
    if(m_ParentModel.m_MLS_degree == 0){
        degree = neighbourhoodSize > 11 ? 3 : neighbourhoodSize > 5 ? 2 : 1;
    } else {
        degree = m_ParentModel.m_MLS_degree;
    }
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
        Eigen::VectorXd x_n(found);
        Eigen::VectorXd y_n(found);
        Eigen::VectorXd z_n(found);
        float sums2[3] = {0,0,0};
        float sums[3] = {0,0,0};
        for (size_t _i = 0; _i < indices->size(); _i++) {
            float _x = m_ParentModel.m_Cloud->at(indices->at(_i)).x;
            x_s[_i] = _x;
            x_n[_i] = m_ParentModel.m_Cloud->at(indices->at(_i)).normal_x;
            sums2[0] += pow(_x,2);
            sums[0] += _x;

            float _y = m_ParentModel.m_Cloud->at(indices->at(_i)).y;
            y_s[_i] = _y;
            y_n[_i] = m_ParentModel.m_Cloud->at(indices->at(_i)).normal_y;
            sums2[1] += pow(_y,2);
            sums[1] += _y;

            float _z = m_ParentModel.m_Cloud->at(indices->at(_i)).z;
            z_s[_i] = _z;
            z_n[_i] = m_ParentModel.m_Cloud->at(indices->at(_i)).normal_z;
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
            distance = (dist2d(gridPoint.x - x_n.mean()/10, _f_2d(coeffs_yz, closest_point.y, closest_point.z), 1) < dist2d(gridPoint.x, _f_2d(coeffs_yz, closest_point.y, closest_point.z), 1) ? 1 : -1 ) *
                           dist2d(gridPoint.x, _f_2d(coeffs_yz, gridPoint.y, gridPoint.z), _closest_k_sqr_distances.at(0));

        } else if (vars[1] < vars[0] && vars[1] < vars[2]) {
            /* y is y */
            /* approximate f(x, z) = y */
            Eigen::MatrixXd mat_xz(found, mat_size);
            for (int _i = 0; _i < found; _i++){
                mat2d(mat_xz, _i, x_s, z_s);
            }
            Eigen::MatrixXd coeffs_xz = pseudoinverse(mat_xz) * y_s;
            distance = (dist2d(gridPoint.y - y_n.mean()/10, _f_2d(coeffs_xz, closest_point.x, closest_point.z), 1) < dist2d(gridPoint.y, _f_2d(coeffs_xz, closest_point.x, closest_point.z), 1) ? 1 : -1 ) *
                        dist2d(gridPoint.y, _f_2d(coeffs_xz, gridPoint.x, gridPoint.z), _closest_k_sqr_distances.at(0));
        } else {
            /* z is y */
            /* approximate f(x, y) = z */
            Eigen::MatrixXd mat_xy(found, mat_size);
            for (int _i = 0; _i < found; _i++){
                mat2d(mat_xy, _i, x_s, y_s);
            }
            Eigen::MatrixXd coeffs_xy = pseudoinverse(mat_xy) * z_s;
            distance = (dist2d(gridPoint.z - z_n.mean()/10, _f_2d(coeffs_xy, closest_point.x, closest_point.y), 1) < dist2d(gridPoint.z, _f_2d(coeffs_xy, closest_point.x, closest_point.y), 1) ? 1 : -1 ) *
                        dist2d(gridPoint.z, _f_2d(coeffs_xy, gridPoint.x, gridPoint.y), _closest_k_sqr_distances.at(0));
        }
        // smallest variance is variable
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


double CloudModel::PCL_HoppeReconstruction() {
    m_Grid.Regenerate();

    ClockGuard timer(__func__);

    // Reconstruction
    pcl::MarchingCubesHoppe<pcl::PointNormal> hoppe;
    hoppe.setInputCloud(m_Cloud);
    hoppe.setPercentageExtendGrid(0.2f);
    hoppe.setGridResolution(m_Grid.GetResX(), m_Grid.GetResY(), m_Grid.GetResZ());
    hoppe.setSearchMethod(m_Tree);
    hoppe.setIsoLevel(m_IsoLevel);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    hoppe.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();

    return timer.ElapsedTime();
}


double CloudModel::PCL_MC_RBF_Reconstruction() {
    m_Grid.Regenerate();

    ClockGuard timer(__func__);

    pcl::MarchingCubesRBF<pcl::PointNormal> rbf;
    rbf.setInputCloud(m_Cloud);
    rbf.setPercentageExtendGrid(0.2f);
    rbf.setGridResolution(m_Grid.GetResX(), m_Grid.GetResY(), m_Grid.GetResZ());
    rbf.setSearchMethod(m_Tree);
    rbf.setIsoLevel(m_IsoLevel);
    rbf.setOffSurfaceDisplacement(m_OffSurfaceDisplacement);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    rbf.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();

    return timer.ElapsedTime();
}


double CloudModel::PCL_PoissonReconstruction() {
    ClockGuard timer(__func__);

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setInputCloud(m_Cloud);
    poisson.setSearchMethod(m_Tree);
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

    return timer.ElapsedTime();
}


double CloudModel::PCL_ConcaveHullReconstruction() {
    ClockGuard timer(__func__);

    pcl::ConcaveHull<pcl::PointNormal> hull;
    hull.setInputCloud(m_Cloud);
    hull.setSearchMethod(m_Tree);
    hull.setAlpha(m_Alpha);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    hull.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();

    return timer.ElapsedTime();
}

double CloudModel::PCL_ConvexHullReconstruction() {
    ClockGuard timer(__func__);

    pcl::ConvexHull<pcl::PointNormal> hull;
    hull.setInputCloud(m_Cloud);
    hull.setSearchMethod(m_Tree);

    std::vector<pcl::Vertices> outputIndices;
    pcl::PointCloud<pcl::PointNormal>::Ptr surfaceCloud(new pcl::PointCloud<pcl::PointNormal>());
    hull.reconstruct(*surfaceCloud, outputIndices);

    ExtractPclReconstructionData(outputIndices, surfaceCloud);
    BufferData();

    return timer.ElapsedTime();
}

double CloudModel::PCL_GP3() {
    ClockGuard timer(__func__);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    gp3.setInputCloud(m_Cloud);
    gp3.setSearchMethod(m_Tree);
    gp3.setNormalConsistency(true);
    gp3.setSearchRadius(m_SearchRadius);
    gp3.setMu(m_Mu);
    gp3.setMaximumAngle(pcl::deg2rad(m_MaxAngle));
    gp3.setMinimumAngle(pcl::deg2rad(m_MinAngle));
    gp3.setMaximumSurfaceAngle(pcl::deg2rad(m_MaxSurfaceAngle));
    gp3.setMaximumNearestNeighbors(m_MaxNN);

    std::vector<pcl::Vertices> outputIndices;
    gp3.reconstruct(outputIndices);

    ExtractPclReconstructionData(outputIndices, m_Cloud);
    BufferData();

    return timer.ElapsedTime();
}

double CloudModel::PCL_OrganizedFastMeshReconstruction() {
    ClockGuard timer(__func__);

    pcl::OrganizedFastMesh<pcl::PointNormal> fastmesh;
    fastmesh.setInputCloud(m_Cloud);
    fastmesh.setSearchMethod(m_Tree);
    fastmesh.setAngleTolerance(m_AngleTolerance);
    fastmesh.setDistanceTolerance(m_DistTolerance);
    fastmesh.setMaxEdgeLength(m_A, m_B, m_C);

    std::vector<pcl::Vertices> outputIndices;
    fastmesh.reconstruct(outputIndices);

    ExtractPclReconstructionData(outputIndices, m_Cloud);
    BufferData();

    return timer.ElapsedTime();
}

ProgramObject CloudModel::s_MeshShader;
ProgramObject CloudModel::s_ShaderProgram;
ProgramObject CloudModel::s_GeometryProgram;
ProgramObject CloudModel::s_ColorProgram;