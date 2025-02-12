#include <cstdio>
#include <iostream>
#include <string>

// 点的类型
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl/impl/point_types.hpp"

// 点云文件IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>

// kd tree
#include <pcl/kdtree/kdtree_flann.h>
#include "pcl/search/kdtree.h"

// 特征提取 
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

// 重构
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/concave_hull.h>

// visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <vtkPLYReader.h>
#include <vtkSTLWriter.h>
#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAxesActor.h>



/// @brief 泊松重建
void Three_D_Reconstruction(std::string& path_read_pcd, std::string& path_save_ply);
/// @brief A-shape重建
void A_Shape_Reconstruction(std::string& path_read_pcd, std::string& path_save_ply, std::string& path_save_pcd, std::string& path_save_obj);
/// @brief PLY转STL
void PLY2STL(std::string& path_read_ply, std::string& path_save_stl);
void visualizeSTL(const std::string& stl_file_path);
double computeAlphaBasedOnDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::search::KdTree<pcl::PointXYZ>& tree);

int main()
{
    std::string path_read_pcd = "../lidar/pcd/RMUC_dense_cut_fan.pcd";
    std::string path_save_ply = "../lidar/mesh/RMUC_dense_cut_fan.ply";
    std::string path_save_pcd = "../lidar/mesh/RMUC_dense_cut_fan.pcd";
    std::string path_save_obj = "../lidar/mesh/RMUC_dense_cut_fan.obj";
    std::string path_save_stl = "../lidar/mesh/RMUC_dense_cut_fan.stl";

    // Three_D_Reconstruction(path_read_pcd, path_save_ply); //效果一坨
    A_Shape_Reconstruction(path_read_pcd, path_save_ply, path_save_pcd, path_save_obj);
    PLY2STL(path_save_ply, path_save_stl);

    return 0;
}


void Three_D_Reconstruction(std::string& path_read_pcd, std::string& path_save_ply)
{
    // read file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(path_read_pcd, *cloud);
    std::cout << "Loaded point cloud size: " << cloud->points.size() << std::endl;
    if (cloud->points.empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
    }


    // 法线估计
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);

    // 连接法线和坐标
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
 
	//___泊松重建___
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);
	//创建Poisson对象，并设置参数
	pcl::Poisson<pcl::PointNormal> pn;
	pn.setSearchMethod(tree2);
	pn.setInputCloud(cloud_with_normals);

	pn.setConfidence(true); //是否使用法向量的大小作为置信信息。如果false，所有法向量均归一化。
	// pn.setDegree(2);   //设置参数degree[1,5],值越大越精细，耗时越久。
	pn.setDepth(6);     // 6 树的最大深度，求解2^d x 2^d x 2^d立方体元。由于八叉树自适应采样密度，指定值仅为最大深度。
	pn.setMinDepth(2); 
	pn.setIsoDivide(6);  // 6 用于提取ISO等值面的算法的深度   
	pn.setSamplesPerNode(10); // 10 设置落入一个八叉树结点中的样本点的最小数量。无噪声，[1.0-5.0],有噪声[15.-20.]平滑
	pn.setScale(1.25); //设置用于重构的立方体直径和样本边界立方体直径的比率。
	pn.setSolverDivide(3); // 3 设置求解线性方程组的Gauss-Seidel迭代方法的深度
	//pn.setIndices();

	pn.setConfidence(false);
	pn.setManifold(false);    //是否添加多边形的重心，当多边形三角化时。
	pn.setOutputPolygons(false);  //是否输出多边形网格（而不是三角化移动立方体的结果）
 
	//设置搜索方法和输入点云
	pn.setManifold(true); //是否添加多边形的重心，当多边形三角化时。 设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
	pn.setOutputPolygons(false); //是否输出多边形网格（而不是三角化移动立方体的结果）
	
	//___保存重建结果___
	//创建多变形网格，用于存储结果
	pcl::PolygonMesh mesh;
	//执行重构
	pn.performReconstruction(mesh);
 
	//保存网格图
	pcl::io::savePLYFile(path_save_ply, mesh);    try {
        pcl::io::savePLYFile(path_save_ply, *cloud);
    } catch (const pcl::IOException& e) {
        std::cerr << "Error saving PLY file: " << e.what() << std::endl;
    }

 
	//___可视化重建结果___
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0, 0, 0);        //  设置背景色为黑色
	viewer->addPolygonMesh(mesh, "my");
	viewer->addCoordinateSystem(1.0);          // 建立空间直角坐标系
	viewer->initCameraParameters();                // 初始化相机参数

	while (!viewer->wasStopped()){
		viewer->spinOnce(100);                                // 显示
		// boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
 
}

double computeAlphaBasedOnDensity(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::search::KdTree<pcl::PointXYZ>& tree) {
    // 计算点云的平均密度
    double total_density = 0.0;
    int point_count = cloud->points.size();
    for (size_t i = 0; i < point_count; ++i) {
        std::vector<int> neighbors;
        std::vector<float> distances;
        tree.radiusSearch(i, 0.1, neighbors, distances); // 以0.1为半径搜索邻居
        total_density += neighbors.size();
    }
    double average_density = total_density / point_count;

    // 根据平均密度计算alpha值
    double alpha = std::max(0.01, 1.0 / average_density); // 防止alpha为负值或过大
    if (alpha < 0.06)
    {
        alpha = 0.06;
    }

    return alpha;
}

void A_Shape_Reconstruction(std::string& path_read_pcd, std::string& path_save_ply, std::string& path_save_pcd, std::string& path_save_obj)
{
    printf("A-shape Reconstruction\n");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path_read_pcd, *cloud) == -1)
    {
        std::cerr << "Error: Unable to load point cloud file!" << std::endl;
        return;
    }
    std::cout << "Loaded point cloud size: " << cloud->points.size() << std::endl;
    if (cloud->points.empty())
    {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        return;
    }

    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> cavehull;
    cavehull.setInputCloud(cloud);
    float alpha = computeAlphaBasedOnDensity(cloud, tree); // 自定义计算函数
    std::cout<<"alpha: "<<alpha<<std::endl;
    cavehull.setAlpha(alpha); // 调整此参数以生成有效的凹包
    std::vector<pcl::Vertices> polygons;
    cavehull.reconstruct(*surface_hull, polygons);

    if (surface_hull->points.empty())
    {
        std::cerr << "Error: Concave hull has no data points!" << std::endl;
        return;
    }

    pcl::PolygonMesh mesh;
    printf("Reconstructing mesh...\n");
    cavehull.reconstruct(mesh);
    if (mesh.polygons.empty())
    {
        std::cerr << "Error: Mesh reconstruction failed!" << std::endl;
        return;
    }

    pcl::io::saveOBJFile(path_save_obj, mesh);
    std::cerr << "Concave hull has: " << surface_hull->points.size() << " data points." << std::endl;

    // 保存网格图
    printf("Saving mesh...\n");
    pcl::io::savePLYFile(path_save_ply, mesh);

    pcl::PCDWriter writer;
    writer.write(path_save_pcd, *surface_hull, false);

    // 可视化重建结果
    printf("Visualizing mesh...\n");
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->setBackgroundColor(0, 0, 0); // 设置背景色为黑色
    viewer->addPolygonMesh(mesh, "my");
    viewer->addCoordinateSystem(1.0); // 建立空间直角坐标系
    viewer->initCameraParameters(); // 初始化相机参数
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100); // 显示
    }

    // 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("hull"));
    viewer2->setWindowName("Alpha-shape 曲面重构");
    viewer2->addPolygonMesh<pcl::PointXYZ>(surface_hull, polygons, "polyline");
    viewer2->spin();
}


void visualizeSTL(const std::string& stl_file_path) {
    // 创建STL文件读取器
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(stl_file_path.c_str());
    reader->Update(); // 更新数据

    // 获取读取到的多边形数据
    vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();

    // 创建PolyData的映射器
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    // 创建演员来显示数据
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // 创建渲染器
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

    // 创建渲染窗口
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("STL Viewer");

    // 创建渲染窗口交互器
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // 设置坐标轴辅助显示
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    vtkSmartPointer<vtkOrientationMarkerWidget> widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    widget->SetOrientationMarker(axes);
    widget->SetInteractor(renderWindowInteractor);
    widget->SetViewport(0.0, 0.0, 0.2, 0.2);
    widget->EnabledOn();

    // 将演员添加到渲染器
    renderer->AddActor(actor);

    // 设置渲染器的背景色
    renderer->SetBackground(0.1, 0.1, 0.1); // 黑色背景

    // 开始渲染
    renderWindow->Render();
    renderWindowInteractor->Start();
}


void PLY2STL(std::string& path_read_ply, std::string& path_save_stl)
{
    pcl::PolygonMesh mesh;
    if (pcl::io::loadPolygonFilePLY(path_read_ply, mesh) == -1)
    {
        PCL_ERROR("Couldn't read the PLY file\n");
        return;
    }

    if (pcl::io::savePolygonFileSTL(path_save_stl, mesh) == -1)
    {
        PCL_ERROR("Couldn't write the STL file\n");
        return;
    }

	// visualizeSTL(path_save_stl);
}


