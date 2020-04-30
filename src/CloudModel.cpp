#include "CloudModel.h"
#include "MCTable.h"


CloudModel::CloudModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, const std::string &shaderDir) :
        m_Cloud(std::move(cloud)),
        m_Tree(new pcl::search::KdTree<pcl::PointNormal>()) {

    m_Tree->setInputCloud(m_Cloud);
    pcl::getMinMax3D(*m_Cloud, m_MinBB, m_MaxBB);
    m_SizeBB = m_MaxBB - m_MinBB;
    m_MinBB -= (m_SizeBB * 0.05f);
    m_MaxBB += (m_SizeBB * 0.05f);
    m_SizeBB = m_MaxBB - m_MinBB;

    printf("[BB] Size: [%f, %f, %f]\n", m_SizeBB.x(), m_SizeBB.y(), m_SizeBB.z());
    printf("[BB] Min: [%f, %f, %f]\n", m_MinBB.x(), m_MinBB.y(), m_MinBB.z());
    printf("[BB] Max: [%f, %f, %f]\n", m_MaxBB.x(), m_MaxBB.y(), m_MaxBB.z());

    if (!s_ShaderProgram) {
        auto basicVS = compileShader(GL_VERTEX_SHADER, Loader::text(shaderDir + "primitive.vert"));
        auto basicFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderDir + "primitive.frag"));
        auto normalsGS = compileShader(GL_GEOMETRY_SHADER, Loader::text(shaderDir + "primitive.geom"));
        s_ShaderProgram = createProgram(basicVS, basicFS);
        s_GeometryProgram = createProgram(basicVS, basicFS, normalsGS);

        auto colorVS = compileShader(GL_VERTEX_SHADER, Loader::text(shaderDir + "wireframe.vert"));
        auto colorFS = compileShader(GL_FRAGMENT_SHADER, Loader::text(shaderDir + "wireframe.frag"));
        s_ColorProgram = createProgram(colorVS, colorFS);
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
    glNamedBufferData(m_VBOs[InputPCNormals], sizeof(glm::vec3) * m_CloudNormals.size(), m_CloudNormals.data(),
                      GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[InputPCNormals], 0);
    glVertexArrayAttribFormat(m_VAOs[InputPCNormals], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[InputPCNormals], 0, m_VBOs[InputPCNormals], 0, sizeof(glm::vec3));

    glNamedBufferData(m_VBOs[InputPC], sizeof(pcl::PointNormal) * m_Cloud->points.size(), m_Cloud->points.data(),
                      GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[InputPC], 0);
    glEnableVertexArrayAttrib(m_VAOs[InputPC], 1);
    glVertexArrayAttribFormat(m_VAOs[InputPC], 0, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, data));
    glVertexArrayAttribFormat(m_VAOs[InputPC], 1, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, normal));
    glVertexArrayVertexBuffer(m_VAOs[InputPC], 0, m_VBOs[InputPC], 0, sizeof(pcl::PointNormal));
    glVertexArrayVertexBuffer(m_VAOs[InputPC], 1, m_VBOs[InputPC], 0, sizeof(pcl::PointNormal));

    // Setup vertex attributes for connection data
    glEnableVertexArrayAttrib(m_VAOs[Connections], 0);
    glVertexArrayAttribFormat(m_VAOs[Connections], 0, 3, GL_FLOAT, GL_FALSE, offsetof(pcl::PointNormal, data));
    glVertexArrayVertexBuffer(m_VAOs[Connections], 0, m_VBOs[Connections], 0, sizeof(pcl::PointNormal));

    // Setup vertex attributes for MC corner data
    glEnableVertexArrayAttrib(m_VAOs[Corners], 0);
    glEnableVertexArrayAttrib(m_VAOs[Corners], 1);
    glVertexArrayAttribFormat(m_VAOs[Corners], 0, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, pos));
    glVertexArrayAttribFormat(m_VAOs[Corners], 1, 3, GL_FLOAT, GL_FALSE, offsetof(VertexRGB, color));
    glVertexArrayVertexBuffer(m_VAOs[Corners], 0, m_VBOs[Corners], 0, sizeof(VertexRGB));
    glVertexArrayVertexBuffer(m_VAOs[Corners], 1, m_VBOs[Corners], 0, sizeof(VertexRGB));

    RegenerateGrid();
}

void CloudModel::Draw(glm::mat4 pv, glm::vec3 color) {
    glm::mat4 modelMatrix(1.0f);
    glm::mat4 pvm = pv * modelMatrix;

    // Draw model
    s_ShaderProgram.use();
    s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
    s_ShaderProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));

    if (m_ShowMesh && !m_MeshVertices.empty()) {
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
        glBindVertexArray(m_VAOs[Model]);
        glDrawArrays(GL_TRIANGLES, 0, m_MeshVertices.size());
    }

    if (m_ShowConnections && !m_Connections.empty()) {
        glm::vec3 connectionColor(1.0f, 0.0f, 0.0f);
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(connectionColor));
        glLineWidth(2.0f);
        glBindVertexArray(m_VAOs[Connections]);
        // glDrawArrays(GL_LINES, m_ConnectionIdx * 2, 2);
        glDrawArrays(GL_LINES, 0, m_Connections.size());
        glLineWidth(1.0f);
    }

    if (m_ShowInputPC) {
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
        glPointSize(3.0f);
        glBindVertexArray(m_VAOs[InputPC]);
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
            glBindVertexArray(m_VAOs[InputPC]);
            glDrawArrays(GL_POINTS, 0, m_Cloud->points.size());
            glLineWidth(1.0f);
        }
    }

    if (m_ShowGrid) {
        // Draw MC corners point cloud
        s_ColorProgram.use();
        s_ColorProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
        glPointSize(3.0f);
        glBindVertexArray(m_VAOs[Corners]);
        glDrawArrays(GL_POINTS, 0, m_CubeCorners.size());
        glPointSize(1.0f);
    }

    //TODO: draw model
    // glBindVertexArray(m_VAOs[Model]);
    // glDrawElements(GL_TRIANGLES, flatIndices.size(), GL_UNSIGNED_INT, nullptr);

    // Draw outlines of model triangles, it looks weird without illumination model otherwise
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
    // glDrawElements(GL_TRIANGLES, flatIndices.size(), GL_UNSIGNED_INT, nullptr);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void CloudModel::RegenerateGrid() {
    size_t cornerIdx = 0;
    float deltaX = m_SizeBB.x() / m_GridSizeX;
    float deltaY = m_SizeBB.y() / m_GridSizeY;
    float deltaZ = m_SizeBB.z() / m_GridSizeZ;
    size_t pointsTotal = (m_GridSizeX + 1) * (m_GridSizeY + 1) * (m_GridSizeZ + 1);
    m_CubeCorners.resize(pointsTotal);
    m_CornersXYZ.resize(pointsTotal);
    m_IsoValues.resize(pointsTotal);

    for (int x = 0; x < m_GridSizeX + 1; ++x) {
        float xPos = m_MinBB.x() + (x * deltaX);
        for (int y = 0; y < m_GridSizeY + 1; ++y) {
            float yPos = m_MinBB.y() + (y * deltaY);
            for (int z = 0; z < m_GridSizeZ + 1; ++z) {
                float zPos = m_MinBB.z() + (z * deltaZ);
                m_CubeCorners[cornerIdx] = VertexRGB(glm::vec3(xPos, yPos, zPos), glm::vec3(0.0f, 0.0f, 1.0f));
                m_CornersXYZ[cornerIdx] = std::make_tuple(x, y, z);
                cornerIdx++;
            }
        }
    }

    // Point cloud of MC corners
    glNamedBufferData(m_VBOs[Corners], sizeof(VertexRGB) * m_CubeCorners.size(), m_CubeCorners.data(), GL_STATIC_DRAW);
    m_InvalidatedGrid = false;
}

void CloudModel::CalculateIsoValues() {
    int K = 1; // Find closest point
    std::vector<int> pointIndices(K);
    std::vector<float> pointDistances(K);
    m_Connections.resize(m_CubeCorners.size() * 2);

    for (size_t i = 0; i < m_CubeCorners.size(); ++i) {
        VertexRGB &cornerVertex(m_CubeCorners[i]);
        pcl::PointNormal cornerPoint;
        cornerPoint.x = cornerVertex.pos.x;
        cornerPoint.y = cornerVertex.pos.y;
        cornerPoint.z = cornerVertex.pos.z;

        int found = m_Tree->nearestKSearch(cornerPoint, K, pointIndices, pointDistances);
        assert(found == K);
        pcl::PointNormal nearestPoint = m_Cloud->points[pointIndices.front()];
        glm::vec3 nearestVertex(nearestPoint.x, nearestPoint.y, nearestPoint.z);
        // Calculate the distance between the MC corner point and the tangent
        // plane of the closest surface point. Dot projection of the point2point
        // vector and the unit-length surface normal gives us the distance.
        auto direction = cornerPoint.getVector3fMap() - nearestPoint.getVector3fMap();
        float distance = nearestPoint.getNormalVector3fMap().dot(direction);

        m_IsoValues[i] = distance;

        // Red color for corner points outside the geometry and green for points that are inside
        cornerVertex.color = distance <= 0 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);

        // size_t x = i % (m_GridSizeX + 1);
        // size_t idx = i / (m_GridSizeX + 1);
        // size_t y = idx % (m_GridSizeY + 1);
        // idx /= (m_GridSizeY + 1);
        // size_t z = idx;
        // printf("[%lu, %lu, %lu] distance: %f\n", x, y, z, distance);

        m_Connections[i * 2] = cornerPoint;
        m_Connections[i * 2 + 1] = nearestPoint;
    }

    // Upload connections data
    glNamedBufferData(m_VBOs[Connections], sizeof(pcl::PointNormal) * m_Connections.size(), nullptr, GL_STATIC_DRAW);

    // Reupload MC corner vertex data with colors
    glNamedBufferSubData(m_VBOs[Corners], 0, sizeof(VertexRGB) * m_CubeCorners.size(), m_CubeCorners.data());
}

void CloudModel::Reconstruct() {
    if (m_InvalidatedGrid) {
        RegenerateGrid();
    }

    CalculateIsoValues();
    m_MeshVertices.clear();

    size_t zPointCount = m_GridSizeZ + 1;
    size_t yzPointCount = (m_GridSizeY + 1) * zPointCount;
    size_t cornerIdx = 0;
    std::array<size_t, 8> cornerOffsets{
            0,
            1,
            zPointCount,
            zPointCount + 1,
            yzPointCount,
            yzPointCount + 1,
            yzPointCount + zPointCount,
            yzPointCount + zPointCount + 1
    };

    for (int x = 0; x < m_GridSizeX + 1; ++x) {
        for (int y = 0; y < m_GridSizeY + 1; ++y) {
            for (int z = 0; z < m_GridSizeZ + 1; ++z) {

                std::array<glm::vec3, 8> corners{
                        m_CubeCorners[cornerIdx].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[1]].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[3]].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[2]].pos,

                        m_CubeCorners[cornerIdx + cornerOffsets[4]].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[5]].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[7]].pos,
                        m_CubeCorners[cornerIdx + cornerOffsets[6]].pos,
                };

                std::array<float, 8> isoValues{
                        m_IsoValues[cornerIdx],
                        m_IsoValues[cornerIdx + cornerOffsets[1]],
                        m_IsoValues[cornerIdx + cornerOffsets[3]],
                        m_IsoValues[cornerIdx + cornerOffsets[2]],

                        m_IsoValues[cornerIdx + cornerOffsets[4]],
                        m_IsoValues[cornerIdx + cornerOffsets[5]],
                        m_IsoValues[cornerIdx + cornerOffsets[7]],
                        m_IsoValues[cornerIdx + cornerOffsets[6]],
                };

                cornerIdx++;

                uint32_t cubeIdx = 0;
                for (size_t i = 0; i < corners.size(); ++i)
                    cubeIdx |= unsigned(isoValues[i] <= 0) << i;

                /* Cube is entirely in/out of the surface */
                if (cubeIdx == 0 || cubeIdx == 255)
                    continue;

                // Find the vertices where the surface intersects the cube
                // TODO: Cache edge intersections to avoid recomputations (future optimization)
                // Maybe map (p1, p2) -> intersection point?
                uint32_t cubeConfig = edgeTable[cubeIdx];
                std::array<glm::vec3, 12> intersections;
                std::array<size_t, 24> vertexIndices{
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

                for (size_t i = 0; i < intersections.size(); ++i) {
                    if (cubeConfig & (1u << i)) {
                        const glm::vec3 &p1(corners[vertexIndices[i * 2]]);
                        const glm::vec3 &p2(corners[vertexIndices[i * 2 + 1]]);
                        intersections[i] = (p1 + p2) / 2.0f;

                         float l0 = isoValues[vertexIndices[i * 2]];
                         float l1 = isoValues[vertexIndices[i * 2 + 1]];
                         const float interpCoeff = (0 - l0) / (l1 - l0);
                         intersections[i] = glm::vec3(
                             p1.x * (1.0f - interpCoeff) + p2.x * interpCoeff,
                             p1.y * (1.0f - interpCoeff) + p2.y * interpCoeff,
                             p1.z * (1.0f - interpCoeff) + p2.z * interpCoeff
                         );
                    }
                }

                // Assemble triangles
                const int8_t *configTriangles = &triangleTable[cubeIdx][0];

                //#pragma omp critical(triangleEmit)
                {
                    for (size_t i = 0; configTriangles[i] != -1; i += 3) {
                        m_MeshVertices.emplace_back(intersections[configTriangles[i]]);
                        m_MeshVertices.emplace_back(intersections[configTriangles[i + 1]]);
                        m_MeshVertices.emplace_back(intersections[configTriangles[i + 2]]);
                    }
                }
            }
        }
    }

    // Buffer model vertex and index data and bind buffers to VAO
    glNamedBufferData(m_VBOs[Model], sizeof(glm::vec3) * m_MeshVertices.size(), m_MeshVertices.data(), GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(m_VAOs[Model], 0);
    glVertexArrayAttribFormat(m_VAOs[Model], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(m_VAOs[Model], 0, m_VBOs[Model], 0, sizeof(glm::vec3));
}

ProgramObject CloudModel::s_ShaderProgram;
ProgramObject CloudModel::s_GeometryProgram;
ProgramObject CloudModel::s_ColorProgram;