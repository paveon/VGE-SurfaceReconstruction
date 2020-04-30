#include "CloudModel.h"
#include "MCTable.h"

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyCloud()
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    for (const auto &bunnyVertex : bunnyVertices)
    {
        pcl::PointNormal pt;
        pt.x = bunnyVertex.position[0];
        pt.y = bunnyVertex.position[1];
        pt.z = bunnyVertex.position[2];
        pt.normal_x = bunnyVertex.normal[0];
        pt.normal_y = bunnyVertex.normal[1];
        pt.normal_z = bunnyVertex.normal[2];
        cloud->push_back(pt);
    }
    return cloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr sphereCloud(float radius)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr sphere(new pcl::PointCloud<pcl::PointNormal>());
    for (float phi = 0; phi < 2 * M_PI; phi += 0.2)
    {
        float z = radius * cos(phi);
        for (float th = 0; th < M_PI; th += 0.2)
        {
            float x = radius * sin(phi) * cos(th);
            float y = radius * sin(phi) * sin(th);
            pcl::PointNormal point;
            glm::vec3 normal = (glm::vec3(x, y, z) - glm::vec3(0.0f, 0.0f, 0.0f));
            normal = glm::normalize(normal);
            point.x = x;
            point.y = y;
            point.z = z;
            point.normal_x = normal.x;
            point.normal_y = normal.y;
            point.normal_z = normal.z;
            sphere->push_back(point);
        }
    }
    sphere->is_dense = true;
    sphere->height = 1;
    sphere->width = sphere->points.size();
    return sphere;
}

CloudModel::CloudModel(const std::string &shaderDir) : m_Cloud(sphereCloud(0.5f)), m_Tree(new pcl::search::KdTree<pcl::PointNormal>())
{
    m_Tree->setInputCloud(m_Cloud);
    pcl::getMinMax3D(*m_Cloud, m_MinBB, m_MaxBB);
    m_SizeBB = m_MaxBB - m_MinBB;
    m_MinBB -= (m_SizeBB * 0.05f);
    m_MaxBB += (m_SizeBB * 0.05f);
    m_SizeBB = m_MaxBB - m_MinBB;

    printf("[BB] Size: [%f, %f, %f]\n", m_SizeBB.x(), m_SizeBB.y(), m_SizeBB.z());
    printf("[BB] Min: [%f, %f, %f]\n", m_MinBB.x(), m_MinBB.y(), m_MinBB.z());
    printf("[BB] Max: [%f, %f, %f]\n", m_MaxBB.x(), m_MaxBB.y(), m_MaxBB.z());

    if (!s_ShaderProgram)
    {
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

    glNamedBufferData(m_VBOs[InputPC], sizeof(pcl::PointNormal) * m_Cloud->points.size() + 4, m_Cloud->points.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(m_VAOs[InputPC], 0, m_VBOs[InputPC], offsetof(pcl::PointNormal, data), sizeof(pcl::PointNormal));
    glVertexArrayVertexBuffer(m_VAOs[InputPC], 1, m_VBOs[InputPC], offsetof(pcl::PointNormal, normal), sizeof(pcl::PointNormal));
    glEnableVertexArrayAttrib(m_VAOs[InputPC], 0);
    glEnableVertexArrayAttrib(m_VAOs[InputPC], 1);

    RegenerateGrid();
}

void CloudModel::Draw(glm::mat4 pv, glm::vec3 color)
{
    glm::mat4 modelMatrix;
    glm::mat4 pvm = pv * modelMatrix;
    // Draw model
    s_ShaderProgram.use();
    s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
    s_ShaderProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));

    if (m_ShowMesh && !m_MeshVertices.empty())
    {
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
        glBindVertexArray(m_VAOs[Model]);
        glDrawArrays(GL_TRIANGLES, 0, m_MeshVertices.size());
    }

    if (m_ShowConnections && m_Connections.size() > 0)
    {
        glm::vec3 connectionColor(1.0f, 0.0f, 0.0f);
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(connectionColor));
        glLineWidth(2.0f);
        glBindVertexArray(m_VAOs[Connections]);
        // glDrawArrays(GL_LINES, m_ConnectionIdx * 2, 2);
        glDrawArrays(GL_LINES, 0, m_Connections.size());
        glLineWidth(1.0f);
    }

    if (m_ShowInputPC)
    {
        s_ShaderProgram.set3fv("primitiveColor", glm::value_ptr(color));
        glPointSize(3.0f);
        glBindVertexArray(m_VAOs[InputPC]);
        glDrawArrays(GL_POINTS, 0, m_Cloud->points.size());
        glPointSize(1.0f);

        if (m_ShowNormals)
        {
            // Draw normals
            glm::vec3 normalColor(1.0f, 0.0f, 1.0f);
            s_GeometryProgram.use();
            s_GeometryProgram.set3fv("primitiveColor", glm::value_ptr(normalColor));
            s_GeometryProgram.setMatrix4fv("pvm", glm::value_ptr(pvm));
            glLineWidth(2.0f);
            glBindVertexArray(m_VAOs[InputPC]);
            glDrawArrays(GL_POINTS, 0, m_Cloud->points.size());
            glLineWidth(1.0f);
        }
    }

    if (m_ShowGrid)
    {
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

void CloudModel::RegenerateGrid()
{
    size_t cornerIdx = 0;
    float deltaX = m_SizeBB.x() / m_GridSizeX;
    float deltaY = m_SizeBB.y() / m_GridSizeY;
    float deltaZ = m_SizeBB.z() / m_GridSizeZ;
    size_t pointsTotal = (m_GridSizeX + 1) * (m_GridSizeY + 1) * (m_GridSizeZ + 1);
    m_CubeCorners.resize(pointsTotal);
    m_IsoValues.resize(pointsTotal);

    for (int x = 0; x < m_GridSizeX + 1; ++x)
    {
        float xPos = m_MinBB.x() + (x * deltaX);
        for (int y = 0; y < m_GridSizeY + 1; ++y)
        {
            float yPos = m_MinBB.y() + (y * deltaY);
            for (int z = 0; z < m_GridSizeZ + 1; ++z)
            {
                float zPos = m_MinBB.z() + (z * deltaZ);
                m_CubeCorners[cornerIdx++] = BasicVertex(glm::vec3(xPos, yPos, zPos), glm::vec3(0.0f, 0.0f, 1.0f));
            }
        }
    }

    // Point cloud of MC corners
    glNamedBufferData(m_VBOs[Corners], sizeof(BasicVertex) * m_CubeCorners.size() + 4, m_CubeCorners.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(m_VAOs[Corners], 0, m_VBOs[Corners], offsetof(BasicVertex, pos), sizeof(BasicVertex));
    glVertexArrayVertexBuffer(m_VAOs[Corners], 1, m_VBOs[Corners], offsetof(BasicVertex, color), sizeof(BasicVertex));
    glEnableVertexArrayAttrib(m_VAOs[Corners], 0);
    glEnableVertexArrayAttrib(m_VAOs[Corners], 1);

    m_InvalidatedGrid = false;
}

void CloudModel::CalculateIsoValues()
{
    int K = 1; // Find closest point
    std::vector<int> pointIndices(K);
    std::vector<float> pointDistances(K);
    m_Connections.resize(m_CubeCorners.size() * 2);

    for (size_t i = 0; i < m_CubeCorners.size(); ++i)
    {
        BasicVertex &cornerVertex(m_CubeCorners[i]);
        pcl::PointNormal cornerPoint;
        cornerPoint.x = cornerVertex.pos.x;
        cornerPoint.y = cornerVertex.pos.y;
        cornerPoint.z = cornerVertex.pos.z;

        int found = m_Tree->nearestKSearch(cornerPoint, K, pointIndices, pointDistances);
        assert(found == K);
        pcl::PointNormal nearestPoint = m_Cloud->points[pointIndices.front()];

        // Calculate the distance between the MC corner point and the tangent
        // plane of the closest surface point. Dot projection of the point2point
        // vector and the unit-length surface normal gives us the distance.
        auto direction = cornerPoint.getVector3fMap() - nearestPoint.getVector3fMap();
        float distance = nearestPoint.getNormalVector3fMap().dot(direction);
        m_IsoValues[i] = distance;

        // Red color for corner points outside the geometry and green for points that are inside
        cornerVertex.color = distance < 0 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);

        // size_t x = i % (m_GridSizeX + 1);
        // size_t idx = i / (m_GridSizeX + 1);
        // size_t y = idx % (m_GridSizeY + 1);
        // idx /= (m_GridSizeY + 1);
        // size_t z = idx;
        // printf("[%lu, %lu, %lu] distance: %f\n", x, y, z, distance);

        m_Connections[i * 2] = cornerPoint;
        m_Connections[i * 2 + 1] = nearestPoint;
    }

    glNamedBufferData(m_VBOs[Connections], sizeof(pcl::PointNormal) * m_Connections.size() + 4, m_Connections.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(m_VAOs[Connections], 0, m_VBOs[Connections], offsetof(pcl::PointNormal, data), sizeof(pcl::PointNormal));
    glEnableVertexArrayAttrib(m_VAOs[Connections], 0);

    // Reupload MC corner vertex data with colors
    glNamedBufferData(m_VBOs[Corners], sizeof(BasicVertex) * m_CubeCorners.size() + 4, m_CubeCorners.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(m_VAOs[Corners], 0, m_VBOs[Corners], offsetof(BasicVertex, pos), sizeof(BasicVertex));
    glVertexArrayVertexBuffer(m_VAOs[Corners], 1, m_VBOs[Corners], offsetof(BasicVertex, color), sizeof(BasicVertex));
    glEnableVertexArrayAttrib(m_VAOs[Corners], 0);
    glEnableVertexArrayAttrib(m_VAOs[Corners], 1);
}

void CloudModel::Reconstruct()
{
    if (m_InvalidatedGrid)
    {
        RegenerateGrid();
    }

    CalculateIsoValues();

    size_t zPointCount = m_GridSizeZ + 1;
    size_t yzPointCount = (m_GridSizeY + 1) * zPointCount;
    size_t cornerIdx = 0;
    std::array<size_t, 8> cornerOffsets{
        0, 1, zPointCount, zPointCount + 1,
        yzPointCount, yzPointCount + 1, yzPointCount + zPointCount, yzPointCount + zPointCount + 1};

    for (int x = 0; x < m_GridSizeX; ++x)
    {
        for (int y = 0; y < m_GridSizeY + 1; ++y)
        {
            for (int z = 0; z < m_GridSizeZ + 1; ++z)
            {
                std::array<glm::vec3, 8> corners{
                    m_CubeCorners[cornerIdx].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[1]].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[2]].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[3]].pos,

                    m_CubeCorners[cornerIdx + cornerOffsets[4]].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[5]].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[6]].pos,
                    m_CubeCorners[cornerIdx + cornerOffsets[7]].pos,
                };

                std::array<float, 8> isoValues{
                    m_IsoValues[cornerIdx],
                    m_IsoValues[cornerIdx + cornerOffsets[1]],
                    m_IsoValues[cornerIdx + cornerOffsets[2]],
                    m_IsoValues[cornerIdx + cornerOffsets[3]],

                    m_IsoValues[cornerIdx + cornerOffsets[4]],
                    m_IsoValues[cornerIdx + cornerOffsets[5]],
                    m_IsoValues[cornerIdx + cornerOffsets[6]],
                    m_IsoValues[cornerIdx + cornerOffsets[7]],
                };

                cornerIdx++;

                uint32_t cubeIdx = 0;
                std::array<bool, 8> verticesEval;
                for (size_t i = 0; i < verticesEval.size(); ++i)
                    cubeIdx |= unsigned(isoValues[i] < 0) << i;

                /* Cube is entirely in/out of the surface */
                if (cubeIdx == 0 || cubeIdx == 255)
                    continue;

                // Find the vertices where the surface intersects the cube
                // TODO: Cache edge intersections to avoid recomputations (future optimization)
                // Maybe map (p1, p2) -> intersection point?
                uint32_t cubeConfig = edgeTable[cubeIdx];
                std::array<glm::vec3, 12> intersections;
                std::array<size_t, 24> vertexIndices{0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};
                for (size_t i = 0; i < intersections.size(); ++i)
                {
                    if (cubeConfig & (1u << i))
                    {
                        const glm::vec3 &p1(corners[vertexIndices[i * 2]]);
                        const glm::vec3 &p2(corners[vertexIndices[i * 2 + 1]]);
                        intersections[i] = (p1 + p2) / 2.0f;

                        // float l0 = isoValues[vertexIndices[i * 2]];
                        // float l1 = isoValues[vertexIndices[i * 2 + 1]];
                        // const float interpCoeff = (0 - l0) / (l1 - l0);
                        // intersections[i] = glm::vec3(
                        //     p1.x * (1.0f - interpCoeff) + p2.x * interpCoeff,
                        //     p1.y * (1.0f - interpCoeff) + p2.y * interpCoeff,
                        //     p1.z * (1.0f - interpCoeff) + p2.z * interpCoeff
                        // );
                    }
                }

                // Assemble triangles
                const int8_t *configTriangles = &triangleTable[cubeIdx][0];

                //#pragma omp critical(triangleEmit)
                {
                    for (size_t i = 0; configTriangles[i] != -1; i += 3)
                    {
                        m_MeshVertices.emplace_back(intersections[configTriangles[i]]);
                        m_MeshVertices.emplace_back(intersections[configTriangles[i + 1]]);
                        m_MeshVertices.emplace_back(intersections[configTriangles[i + 2]]);
                    }
                }
            }
        }
    }

    // Buffer model vertex and index data and bind buffers to VAO
    glNamedBufferData(m_VBOs[Model], sizeof(glm::vec3) * m_MeshVertices.size() + 4, m_MeshVertices.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(m_VAOs[Model], 0, m_VBOs[Model], 0, sizeof(glm::vec3));
    glEnableVertexArrayAttrib(m_VAOs[Model], 0);
}

ProgramObject CloudModel::s_ShaderProgram;
ProgramObject CloudModel::s_GeometryProgram;
ProgramObject CloudModel::s_ColorProgram;