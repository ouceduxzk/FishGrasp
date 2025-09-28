#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示LabelMe格式JSON文件的处理
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from process_bbox_landmark_data import BboxLandmarkProcessor


def create_sample_labelme_json():
    """创建示例LabelMe格式的JSON文件"""
    sample_data = {
        "version": "5.8.3",
        "flags": {},
        "shapes": [
            {
                "label": "头",
                "points": [[1583.6190476190477, 512.2857142857142]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            },
            {
                "label": "身体",
                "points": [[1188.3809523809523, 1531.3333333333333]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            },
            {
                "label": "尾部",
                "points": [[874.0952380952381, 2712.285714285714]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            },
            {
                "label": "鱿鱼",
                "points": [
                    [583.6190476190477, 3693.2380952380954],
                    [1731.2380952380954, 317.047619047619]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": "鱿鱼 (1).jpg",
        "imageData": None,
        "imageHeight": 4096,
        "imageWidth": 3072
    }
    
    return sample_data


def demo_labelme_processing():
    """演示LabelMe格式处理"""
    print("🎯 LabelMe格式处理演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_annotation = create_sample_labelme_json()
    
    print("📄 示例LabelMe格式JSON:")
    print(json.dumps(sample_annotation, indent=2, ensure_ascii=False))
    
    # 创建处理器
    processor = BboxLandmarkProcessor("./demo_data", "./demo_output")
    
    # 提取关键点
    print("\n🔍 提取关键点:")
    landmarks = processor.extract_landmarks_from_annotation(sample_annotation)
    for name, coord in landmarks:
        print(f"  {name}: ({coord[0]:.1f}, {coord[1]:.1f})")
    
    # 提取边界框
    print("\n📦 提取边界框:")
    bbox = processor.extract_bbox_from_annotation(sample_annotation)
    if bbox:
        x1, y1, x2, y2 = bbox
        print(f"  边界框: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
        print(f"  宽度: {x2-x1:.1f}, 高度: {y2-y1:.1f}")
    else:
        print("  ❌ 未找到边界框")
    
    print("\n✅ 演示完成！")


def demo_original_format():
    """演示原始格式处理"""
    print("\n🎯 原始格式处理演示")
    print("=" * 50)
    
    # 创建原始格式数据
    original_annotation = {
        "头部": [100, 50],
        "身体": [100, 150],
        "bbox": [50, 25, 150, 175]
    }
    
    print("📄 示例原始格式JSON:")
    print(json.dumps(original_annotation, indent=2, ensure_ascii=False))
    
    # 创建处理器
    processor = BboxLandmarkProcessor("./demo_data", "./demo_output")
    
    # 提取关键点
    print("\n🔍 提取关键点:")
    landmarks = processor.extract_landmarks_from_annotation(original_annotation)
    for name, coord in landmarks:
        print(f"  {name}: ({coord[0]}, {coord[1]})")
    
    # 提取边界框
    print("\n📦 提取边界框:")
    bbox = processor.extract_bbox_from_annotation(original_annotation)
    if bbox:
        x1, y1, x2, y2 = bbox
        print(f"  边界框: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"  宽度: {x2-x1}, 高度: {y2-y1}")
    else:
        print("  ❌ 未找到边界框")
    
    print("\n✅ 演示完成！")


def main():
    """主函数"""
    print("🚀 BboxLandmarkProcessor LabelMe格式支持演示")
    print("=" * 60)
    
    try:
        demo_labelme_processing()
        demo_original_format()
        
        print("\n🎉 所有演示完成！")
        print("\n💡 提示:")
        print("  - 现在支持LabelMe格式的JSON标注文件")
        print("  - 支持头部、身体、尾部三个关键点")
        print("  - 向后兼容原始格式的JSON文件")
        print("  - 可以处理包含边界框和关键点的复杂标注")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
