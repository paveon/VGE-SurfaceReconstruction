#define SDL_MAIN_HANDLED

#include <iostream>
#include <thread>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/poisson.h>
#include "bunny.h"

pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

void show(const pcl::PolygonMesh &mesh);

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithNormals();
pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithOutNormals();

int main(int /*argc*/, char** /*argv*/) {
   /***************** SAMPLE CODE FOR PCL *****************/
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals;
    if (false)
        cloud_point_normals= bunnyWithOutNormals();
    else
        cloud_point_normals= bunnyWithNormals();

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(12);
    poisson.setInputCloud(cloud_point_normals);
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);
    show(mesh);

    /***************** SAMPLE CODE FOR PCL *****************/
}

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithNormals() {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals(new pcl::PointCloud<pcl::PointNormal>());
    for (const auto & bunnyVertice : bunnyVertices){
        auto *pt = new pcl::PointNormal;
        pt->x = bunnyVertice.position[0];
        pt->y = bunnyVertice.position[1];
        pt->z = bunnyVertice.position[2];
        pt->normal_x = bunnyVertice.normal[0];
        pt->normal_y = bunnyVertice.normal[1];
        pt->normal_z = bunnyVertice.normal[2];
        cloud_point_normals->push_back(*pt);
    }
    return cloud_point_normals;
}

pcl::PointCloud<pcl::PointNormal>::Ptr bunnyWithOutNormals() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_point(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto & bunnyVertice : bunnyVertices){
        auto *pt = new pcl::PointXYZ;
        pt->x = bunnyVertice.position[0];
        pt->y = bunnyVertice.position[1];
        pt->z = bunnyVertice.position[2];
        cloud_point->push_back(*pt);
    }
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud (cloud_point);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    normalEstimation.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    // Use all neighbors in a sphere of radius 3cm
    normalEstimation.setRadiusSearch (0.03);
    // Compute the features
    normalEstimation.compute (*cloud_normals);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_point_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud_point, *cloud_normals, *cloud_point_normals);
    std::cout << cloud_point_normals->size() << std::endl;
    return cloud_point_normals;
}

void show(const pcl::PolygonMesh &mesh) {
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->setCameraPosition(0,0,0,0,0,0);
    viewer->addPolygonMesh(mesh);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> *ne = new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>;
    ne->setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne->setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne->setRadiusSearch (0.3);

    // Compute the features
    ne->compute (*cloud_normals);
    return cloud_normals;
}
