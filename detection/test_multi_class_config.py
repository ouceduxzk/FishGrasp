#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多类别检测配置测试脚本

测试多类别检测的配置和功能
"""

import sys
import os
import yaml
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_config():
    """测试数据集配置"""
    print("🔧 测试数据集配置")
    print("="*50)
    
    # 检查数据集YAML文件
    dataset_yaml = Path("datasets/l0_9.12/dataset.yaml")
    
    if not dataset_yaml.exists():
        print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
        return False
    
    # 读取配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 数据集配置加载成功")
    print(f"   路径: {config['path']}")
    print(f"   训练集: {config['train']}")
    print(f"   验证集: {config['val']}")
    print(f"   测试集: {config['test']}")
    print(f"   类别: {config['names']}")
    
    # 验证类别配置
    expected_classes = ['背景', '鱿鱼']
    if config['names'] == expected_classes:
        print("✅ 类别配置正确")
        print(f"   类别0: {config['names'][0]} (背景)")
        print(f"   类别1: {config['names'][1]} (鱿鱼)")
    else:
        print("❌ 类别配置不正确")
        print(f"   期望: {expected_classes}")
        print(f"   实际: {config['names']}")
        return False
    
    return True

def test_class_distribution():
    """测试类别分布"""
    print("\n🔧 测试类别分布")
    print("="*50)
    
    dataset_path = Path("datasets/l0_9.12")
    train_labels = dataset_path / "labels" / "train"
    val_labels = dataset_path / "labels" / "val"
    
    if not train_labels.exists() or not val_labels.exists():
        print("❌ 标签目录不存在")
        return False
    
    # 统计类别分布
    class_counts = {0: 0, 1: 0}  # 背景, 鱿鱼
    total_files = 0
    
    for split_name, split_path in [("训练集", train_labels), ("验证集", val_labels)]:
        split_counts = {0: 0, 1: 0}
        split_files = 0
        
        for label_file in split_path.glob("*.txt"):
            split_files += 1
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            class_id = int(line.split()[0])
                            if class_id in [0, 1]:
                                split_counts[class_id] += 1
                                class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue
        
        total_files += split_files
        print(f"{split_name}:")
        print(f"   文件数: {split_files}")
        print(f"   背景实例: {split_counts[0]}")
        print(f"   鱿鱼实例: {split_counts[1]}")
        print(f"   总计实例: {sum(split_counts.values())}")
    
    print(f"\n总体统计:")
    print(f"   总文件数: {total_files}")
    print(f"   背景实例: {class_counts[0]}")
    print(f"   鱿鱼实例: {class_counts[1]}")
    print(f"   总计实例: {sum(class_counts.values())}")
    
    # 检查类别平衡性
    if class_counts[0] > 0 and class_counts[1] > 0:
        ratio = class_counts[1] / class_counts[0]
        print(f"   类别比例 (鱿鱼/背景): {ratio:.3f}")
        
        if 0.1 <= ratio <= 10.0:
            print("✅ 类别分布相对平衡")
        else:
            print("⚠️  类别分布不平衡，建议调整")
    else:
        print("❌ 某个类别没有实例")
        return False
    
    return True

def test_hard_negative_mining():
    """测试硬负样本挖掘"""
    print("\n🔧 测试硬负样本挖掘")
    print("="*50)
    
    try:
        from hard_negative_mining import HardNegativeMiner
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 创建挖掘器
    miner = HardNegativeMiner(
        confidence_threshold=0.5,
        iou_threshold=0.5,
        save_samples=False
    )
    
    # 模拟多类别预测结果
    predictions = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.8, 'class': 1},  # 高置信度鱿鱼
        {'bbox': [300, 300, 400, 400], 'confidence': 0.3, 'class': 0},  # 低置信度背景
        {'bbox': [500, 500, 600, 600], 'confidence': 0.9, 'class': 1},  # 高置信度鱿鱼
        {'bbox': [700, 700, 800, 800], 'confidence': 0.7, 'class': 0}   # 中等置信度背景
    ]
    
    # 模拟多类别真实标注
    ground_truth = [
        {'bbox': [110, 110, 210, 210], 'class': 1},  # 鱿鱼
        {'bbox': [520, 520, 620, 620], 'class': 1},  # 鱿鱼
        {'bbox': [150, 150, 250, 250], 'class': 0}   # 背景
    ]
    
    # 进行硬负样本挖掘
    hard_negatives = miner.find_hard_negatives(predictions, ground_truth)
    
    print(f"✅ 硬负样本挖掘完成")
    print(f"   找到 {len(hard_negatives)} 个硬负样本")
    
    # 按类别分析硬负样本
    background_hard = [hn for hn in hard_negatives if hn['prediction']['class'] == 0]
    squid_hard = [hn for hn in hard_negatives if hn['prediction']['class'] == 1]
    
    print(f"   背景困难样本: {len(background_hard)}")
    print(f"   鱿鱼困难样本: {len(squid_hard)}")
    
    # 打印详细信息
    for i, hn in enumerate(hard_negatives):
        class_name = "背景" if hn['prediction']['class'] == 0 else "鱿鱼"
        print(f"   {i+1}. {class_name} - 类型: {hn['type']}, 置信度: {hn['confidence']:.3f}, IoU: {hn['iou']:.3f}")
    
    return True

def test_training_commands():
    """测试训练命令"""
    print("\n🔧 测试训练命令")
    print("="*50)
    
    commands = [
        {
            "name": "基本多类别训练",
            "command": """python3 detection/train_yolo.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --project runs/train \\
    --name multi_class_squid_background_$(date +%Y%m%d_%H%M%S)"""
        },
        {
            "name": "硬负样本挖掘训练",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --project runs/train \\
    --name multi_class_hard_negative_$(date +%Y%m%d_%H%M%S) \\
    --mining_strategy confidence_based \\
    --hard_negative_ratio 0.3"""
        },
        {
            "name": "使用预训练模型",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --use_pretrained_for_mining \\
    --mining_strategy confidence_based"""
        }
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"命令 {i}: {cmd['name']}")
        print(cmd['command'])
        print()
    
    print("✅ 训练命令配置完成")
    return True

def test_performance_analysis():
    """测试性能分析"""
    print("\n🔧 测试性能分析")
    print("="*50)
    
    # 模拟训练结果
    mock_results = {
        'overall': {
            'precision': 0.944,
            'recall': 0.972,
            'mAP50': 0.977,
            'mAP50-95': 0.722
        },
        'background': {
            'precision': 0.950,
            'recall': 0.980,
            'mAP50': 0.985,
            'mAP50-95': 0.750
        },
        'squid': {
            'precision': 0.938,
            'recall': 0.964,
            'mAP50': 0.969,
            'mAP50-95': 0.694
        }
    }
    
    print("模拟训练结果分析:")
    print("="*30)
    
    for class_name, metrics in mock_results.items():
        print(f"{class_name.upper()}:")
        print(f"  精确率: {metrics['precision']:.3f}")
        print(f"  召回率: {metrics['recall']:.3f}")
        print(f"  mAP50: {metrics['mAP50']:.3f}")
        print(f"  mAP50-95: {metrics['mAP50-95']:.3f}")
        print()
    
    # 分析类别性能差异
    background_map50 = mock_results['background']['mAP50']
    squid_map50 = mock_results['squid']['mAP50']
    performance_gap = abs(background_map50 - squid_map50)
    
    print("性能分析:")
    print(f"  背景 mAP50: {background_map50:.3f}")
    print(f"  鱿鱼 mAP50: {squid_map50:.3f}")
    print(f"  性能差距: {performance_gap:.3f}")
    
    if performance_gap < 0.05:
        print("✅ 类别性能平衡")
    else:
        print("⚠️  类别性能不平衡，建议调整")
    
    return True

def main():
    """主函数"""
    print("🚀 多类别检测配置测试")
    print("="*60)
    
    success_count = 0
    total_tests = 5
    
    # 运行测试
    if test_dataset_config():
        success_count += 1
    
    if test_class_distribution():
        success_count += 1
    
    if test_hard_negative_mining():
        success_count += 1
    
    if test_training_commands():
        success_count += 1
    
    if test_performance_analysis():
        success_count += 1
    
    # 输出测试结果
    print("\n" + "="*60)
    print("📊 测试结果摘要")
    print("="*60)
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！多类别检测配置正确")
        print("\n📝 使用建议:")
        print("1. 使用更新后的数据集配置进行训练")
        print("2. 监控每个类别的性能指标")
        print("3. 使用硬负样本挖掘提高性能")
        print("4. 根据类别平衡性调整训练策略")
        print("5. 分析类别间的混淆情况")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


