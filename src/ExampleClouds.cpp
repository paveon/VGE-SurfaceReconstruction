#include "ExampleClouds.h"

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include "bunny.h"

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyCloud() {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    for (const auto &bunnyVertex : bunnyVertices) {
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

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyCloudEstimatedNormals() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto &bunnyVertex : bunnyVertices) {
        pcl::PointXYZ point;
        point.x = bunnyVertex.position[0];
        point.y = bunnyVertex.position[1];
        point.z = bunnyVertex.position[2];
        cloudPoints->push_back(point);
    }

    // TODO: probably need to set viewpoint because default is (0,0,0)?
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);

    normalEstimation.setInputCloud(cloudPoints);
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(0.03); // Use all neighbors in a sphere of radius 3cm
    normalEstimation.compute(*cloudNormals); // Compute the features

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloudPoints, *cloudNormals, *cloud);

    return cloud;
}


pcl::PointCloud<pcl::PointNormal>::Ptr sphereCloud(float radius) {
    pcl::PointCloud<pcl::PointNormal>::Ptr sphere(new pcl::PointCloud<pcl::PointNormal>());
    float deltaPhi = 0.2f;
    float deltaTheta = 0.2f;
    size_t stepsPhi = (2 * M_PI) / deltaPhi;
    size_t stepsTheta = (2 * M_PI) / deltaTheta;
    for (size_t i = 0; i < stepsPhi; i++) {
        float phi = i * deltaPhi;
        float z = radius * std::cos(phi);
        for (size_t j = 0; j < stepsTheta; j++) {
            float theta = j * deltaTheta;
            float x = radius * std::sin(phi) * std::cos(theta);
            float y = radius * std::sin(phi) * std::sin(theta);
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