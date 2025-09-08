#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云工具模块 - 提供点云处理和法向量估计功能

命令行用法:
    python point_cloud_utils.py input.ply -o output.ply -m open3d -k 20 -v
"""

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import argparse
import os
import sys
from datetime import datetime


class PointCloudUtils:
    """点云工具类"""
    
    def __init__(self):
        """初始化点云工具类"""
        pass
    
    @staticmethod
    def estimate_normals_pca(points, k_neighbors=10, radius=None):
        """使用PCA方法估计点云法向量"""
        if len(points) < 3:
            print("警告：点云点数太少，无法估计法向量")
            return None, None
        
        points = np.asarray(points, dtype=np.float64)
        N = len(points)
        normals = np.zeros((N, 3))
        curvatures = np.zeros(N)
        
        # 构建KD树进行近邻搜索
        if radius is not None:
            tree = NearestNeighbors(radius=radius).fit(points)
            neighbors_list = tree.radius_neighbors(points, return_distance=False)
        else:
            tree = NearestNeighbors(n_neighbors=min(k_neighbors, N)).fit(points)
            neighbors_list = tree.kneighbors(points, return_distance=False)
        
        for i in range(N):
            neighbors = neighbors_list[i]
            if len(neighbors) < 3:
                neighbors = np.arange(N)
            
            # 计算局部协方差矩阵
            local_points = points[neighbors]
            centroid = np.mean(local_points, axis=0)
            centered_points = local_points - centroid
            
            # 计算协方差矩阵
            cov_matrix = np.cov(centered_points.T)
            
            # 特征值分解
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                normal = eigenvectors[:, 0]
                
                # 确保法向量指向外部
                global_center = np.mean(points, axis=0)
                if np.dot(normal, centroid - global_center) < 0:
                    normal = -normal
                
                normals[i] = normal
                
                # 计算曲率
                if np.sum(eigenvalues) > 0:
                    curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
                else:
                    curvatures[i] = 0
                    
            except np.linalg.LinAlgError:
                normals[i] = [0, 0, 1]
                curvatures[i] = 0
        
        return normals, curvatures
    
    @staticmethod
    def estimate_normal_for_point(points, point_id, k_neighbors=10, radius=None):
        """计算指定点ID的法向量"""
        if len(points) < 3:
            print("警告：点云点数太少，无法估计法向量")
            return None, None
        
        points = np.asarray(points, dtype=np.float64)
        N = len(points)
        
        if point_id < 0 or point_id >= N:
            print(f"错误：点ID {point_id} 超出范围 [0, {N-1}]")
            return None, None
        
        # 构建KD树进行近邻搜索
        if radius is not None:
            tree = NearestNeighbors(radius=radius).fit(points)
            neighbors = tree.radius_neighbors([points[point_id]], return_distance=False)[0]
        else:
            tree = NearestNeighbors(n_neighbors=min(k_neighbors, N)).fit(points)
            neighbors = tree.kneighbors([points[point_id]], return_distance=False)[0]
        
        if len(neighbors) < 3:
            neighbors = np.arange(N)
        
        # 计算局部协方差矩阵
        local_points = points[neighbors]
        centroid = np.mean(local_points, axis=0)
        centered_points = local_points - centroid
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        
        # 特征值分解
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0]
            
            # 确保法向量指向外部
            global_center = np.mean(points, axis=0)
            if np.dot(normal, centroid - global_center) < 0:
                normal = -normal
            
            # 计算曲率
            if np.sum(eigenvalues) > 0:
                curvature = eigenvalues[0] / np.sum(eigenvalues)
            else:
                curvature = 0
                
            print(f"点 {point_id} 的法向量: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
            print(f"点 {point_id} 的曲率: {curvature:.6f}")
            print(f"点 {point_id} 的坐标: [{points[point_id][0]:.6f}, {points[point_id][1]:.6f}, {points[point_id][2]:.6f}]")
            
            return normal, curvature
            
        except np.linalg.LinAlgError:
            print(f"警告：点 {point_id} 的法向量计算失败，使用默认法向量")
            return np.array([0, 0, 1]), 0
    
    @staticmethod
    def estimate_normals_open3d(points, k_neighbors=10, radius=None):
        """使用Open3D估计点云法向量"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if radius is not None:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
                )
            else:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
                )
            
            normals = np.asarray(pcd.normals)
            curvatures = PointCloudUtils.estimate_curvature_from_normals(points, normals)
            
            return normals, curvatures
            
        except Exception as e:
            print(f"Open3D法向量估计失败: {e}")
            return PointCloudUtils.estimate_normals_pca(points, k_neighbors, radius)
    
    @staticmethod
    def estimate_curvature_from_normals(points, normals, k_neighbors=10):
        """从法向量计算曲率"""
        if len(points) < 3:
            return np.zeros(len(points))
        
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        N = len(points)
        curvatures = np.zeros(N)
        
        tree = NearestNeighbors(n_neighbors=min(k_neighbors, N)).fit(points)
        neighbors_list = tree.kneighbors(points, return_distance=False)
        
        for i in range(N):
            neighbors = neighbors_list[i]
            if len(neighbors) < 3:
                continue
            
            local_normals = normals[neighbors]
            normal_variations = np.std(local_normals, axis=0)
            curvatures[i] = np.linalg.norm(normal_variations)
        
        return curvatures
    
    @staticmethod
    def smooth_normals(points, normals, k_neighbors=10, smoothing_factor=0.5):
        """平滑法向量"""
        if len(points) < 3:
            return normals
        
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        N = len(points)
        smoothed_normals = normals.copy()
        
        tree = NearestNeighbors(n_neighbors=min(k_neighbors, N)).fit(points)
        neighbors_list = tree.kneighbors(points, return_distance=False)
        
        for i in range(N):
            neighbors = neighbors_list[i]
            if len(neighbors) < 2:
                continue
            
            local_normals = normals[neighbors]
            weights = np.exp(-np.linalg.norm(points[neighbors] - points[i], axis=1))
            weights = weights / np.sum(weights)
            
            weighted_normal = np.average(local_normals, axis=0, weights=weights)
            weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)
            
            smoothed_normals[i] = (1 - smoothing_factor) * normals[i] + smoothing_factor * weighted_normal
            smoothed_normals[i] = smoothed_normals[i] / np.linalg.norm(smoothed_normals[i])
        
        return smoothed_normals
    
    @staticmethod
    def load_pointcloud(filename):
        """加载点云文件"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.ply', '.pcd']:
                pcd = o3d.io.read_point_cloud(filename)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                
            elif file_ext in ['.xyz', '.txt']:
                data = np.loadtxt(filename, delimiter=' ', skiprows=0)
                if data.shape[1] >= 3:
                    points = data[:, :3]
                    colors = data[:, 3:6] if data.shape[1] >= 6 else None
                else:
                    raise ValueError(f"文件格式错误：期望至少3列，实际{data.shape[1]}列")
                    
            elif file_ext == '.npy':
                data = np.load(filename)
                if data.shape[1] >= 3:
                    points = data[:, :3]
                    colors = data[:, 3:6] if data.shape[1] >= 6 else None
                else:
                    raise ValueError(f"文件格式错误：期望至少3列，实际{data.shape[1]}列")
                    
            else:
                raise ValueError(f"不支持的文件格式：{file_ext}")
            
            print(f"成功加载点云：{len(points)} 个点")
            if colors is not None:
                print(f"包含颜色信息：{colors.shape}")
            
            return points, colors
            
        except Exception as e:
            print(f"加载点云文件失败: {e}")
            return None, None
    
    @staticmethod
    def save_pointcloud_with_normals(points, normals, colors=None, filename=None, 
                                   format='ply', include_curvature=False, curvatures=None):
        """保存带法向量的点云文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_with_normals_{timestamp}.{format}"
        
        try:
            if format == 'ply':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                success = o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
                if success:
                    print(f"点云已保存为PLY格式: {filename}")
                else:
                    print("保存PLY文件失败")
                    
            elif format == 'xyz':
                if colors is not None:
                    data = np.column_stack([points, normals, colors])
                    header = 'X Y Z NX NY NZ R G B'
                else:
                    data = np.column_stack([points, normals])
                    header = 'X Y Z NX NY NZ'
                
                if include_curvature and curvatures is not None:
                    data = np.column_stack([data, curvatures])
                    header += ' CURVATURE'
                
                np.savetxt(filename, data, header=header, fmt='%.6f')
                print(f"点云已保存为XYZ格式: {filename}")
                
            elif format == 'npy':
                if colors is not None:
                    data = np.column_stack([points, normals, colors])
                else:
                    data = np.column_stack([points, normals])
                
                if include_curvature and curvatures is not None:
                    data = np.column_stack([data, curvatures])
                
                np.save(filename, data)
                print(f"点云已保存为numpy格式: {filename}")
                
            else:
                raise ValueError(f"不支持的输出格式: {format}")
                
        except Exception as e:
            print(f"保存点云文件失败: {e}")
    
    @staticmethod
    def process_pointcloud_file(input_file, output_file=None, method='pca', 
                               k_neighbors=15, radius=None, smooth_factor=0.3,
                               include_curvature=True, visualize=False, format='ply'):
        """处理点云文件：加载、估计法向量、保存"""
        print(f"开始处理点云文件: {input_file}")
        
        # 1. 加载点云
        points, colors = PointCloudUtils.load_pointcloud(input_file)
        if points is None:
            return False
        
        # 2. 估计法向量
        print(f"使用{method}方法估计法向量...")
        if method == 'pca':
            normals, curvatures = PointCloudUtils.estimate_normals_pca(
                points, k_neighbors, radius
            )
        elif method == 'open3d':
            normals, curvatures = PointCloudUtils.estimate_normals_open3d(
                points, k_neighbors, radius
            )
        else:
            print(f"未知的方法: {method}，使用PCA方法")
            normals, curvatures = PointCloudUtils.estimate_normals_pca(
                points, k_neighbors, radius
            )
        
        if normals is None:
            print("法向量估计失败")
            return False
        
        # 3. 平滑法向量
        if smooth_factor > 0:
            print(f"平滑法向量 (因子: {smooth_factor})...")
            normals = PointCloudUtils.smooth_normals(
                points, normals, k_neighbors, smooth_factor
            )
        
        # 4. 保存结果
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_with_normals.{format}"
        
        PointCloudUtils.save_pointcloud_with_normals(
            points, normals, colors, output_file, format, include_curvature, curvatures
        )
        
        print(f"处理完成！输出文件: {output_file}")
        return True


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="点云法向量估计工具",
        epilog="""
使用示例:
  python point_cloud_utils.py input.ply
  python point_cloud_utils.py input.ply -o output.ply -m open3d -k 20
  python point_cloud_utils.py input.ply -o output.xyz -f xyz -c
  python point_cloud_utils.py input.ply -p 100 -k 15  # 只计算点100的法向量
        """
    )
    
    parser.add_argument('input_file', help='输入点云文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（可选）')
    parser.add_argument('-m', '--method', choices=['pca', 'open3d'], default='pca',
                       help='法向量估计方法 (默认: pca)')
    parser.add_argument('-k', '--k_neighbors', type=int, default=15,
                       help='近邻点数量 (默认: 15)')
    parser.add_argument('-r', '--radius', type=float, help='搜索半径')
    parser.add_argument('-s', '--smooth', type=float, default=0.3,
                       help='平滑因子 0-1 (默认: 0.3)')
    parser.add_argument('-f', '--format', choices=['ply', 'xyz', 'npy'], default='ply',
                       help='输出格式 (默认: ply)')
    parser.add_argument('-c', '--curvature', action='store_true',
                       help='包含曲率信息')
    parser.add_argument('-p', '--point_id', type=int, help='只计算指定点ID的法向量（0-based索引）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 如果指定了点ID，只计算该点的法向量
    if args.point_id is not None:
        print(f"计算点 {args.point_id} 的法向量...")
        
        # 加载点云
        points, colors = PointCloudUtils.load_pointcloud(args.input_file)
        if points is None:
            print("❌ 点云加载失败！")
            sys.exit(1)
        
        # 计算指定点的法向量
        normal, curvature = PointCloudUtils.estimate_normal_for_point(
            points, args.point_id, args.k_neighbors, args.radius
        )
        
        if normal is not None:
            print("✅ 单点法向量计算成功完成！")
            sys.exit(0)
        else:
            print("❌ 单点法向量计算失败！")
            sys.exit(1)
    else:
        # 处理整个点云
        success = PointCloudUtils.process_pointcloud_file(
            input_file=args.input_file,
            output_file=args.output,
            method=args.method,
            k_neighbors=args.k_neighbors,
            radius=args.radius,
            smooth_factor=args.smooth,
            include_curvature=args.curvature,
            format=args.format
        )
        
        if success:
            print("✅ 点云处理成功完成！")
            sys.exit(0)
        else:
            print("❌ 点云处理失败！")
            sys.exit(1)


if __name__ == "__main__":
    main()
