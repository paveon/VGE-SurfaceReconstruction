#ifndef EXAMPLE_CLOUDS_H
#define EXAMPLE_CLOUDS_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyCloud();

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyCloudEstimatedNormals();

pcl::PointCloud<pcl::PointNormal>::Ptr sphereCloud(float radius);

#endif