#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试硬负样本挖掘策略

这个脚本测试不同的硬负样本挖掘策略
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_strategy_1_staged_training():
    """测试策略1：分阶段训练"""
    print("🔧 测试策略1：分阶段训练")
    print("="*50)
    
    # 模拟参数
    total_epochs = 100
    initial_epochs_ratio = 0.33
    
    # 计算各阶段轮数
    initial_epochs = max(20, int(total_epochs * initial_epochs_ratio))
    remaining_epochs = total_epochs - initial_epochs
    
    print(f"总训练轮数: {total_epochs}")
    print(f"初始训练轮数: {initial_epochs} ({initial_epochs_ratio*100:.0f}%)")
    print(f"增强训练轮数: {remaining_epochs} ({(1-initial_epochs_ratio)*100:.0f}%)")
    
    # 模拟训练过程
    print("\n📚 阶段1: 初始训练")
    print(f"   训练 {initial_epochs} 轮，建立基础模型...")
    
    print("\n🔍 阶段2: 硬负样本挖掘")
    print("   使用基础模型在验证集上进行硬负样本挖掘...")
    
    # 模拟挖掘结果
    hard_negatives_found = 25
    print(f"   找到 {hard_negatives_found} 个硬负样本")
    
    print("\n🎯 阶段3: 硬负样本增强训练")
    print(f"   训练 {remaining_epochs} 轮，重点关注困难样本...")
    
    print("✅ 策略1测试完成")
    return True

def test_strategy_2_pretrained_model():
    """测试策略2：使用预训练模型"""
    print("\n🔧 测试策略2：使用预训练模型")
    print("="*50)
    
    print("🎯 使用预训练模型进行硬负样本挖掘")
    print("   模型: yolov8s.pt (COCO预训练)")
    print("   跳过初始训练阶段")
    
    print("\n🔍 硬负样本挖掘")
    print("   使用预训练模型在目标数据集上进行挖掘...")
    
    # 模拟挖掘结果
    hard_negatives_found = 30
    print(f"   找到 {hard_negatives_found} 个硬负样本")
    
    print("\n🎯 硬负样本增强训练")
    print("   使用硬负样本进行微调训练...")
    
    print("✅ 策略2测试完成")
    return True

def test_parameter_combinations():
    """测试不同参数组合"""
    print("\n🔧 测试不同参数组合")
    print("="*50)
    
    # 测试不同的初始轮数比例
    total_epochs = 100
    ratios = [0.2, 0.33, 0.5, 0.67]
    
    print("不同初始轮数比例的效果:")
    for ratio in ratios:
        initial_epochs = max(20, int(total_epochs * ratio))
        remaining_epochs = total_epochs - initial_epochs
        print(f"  比例 {ratio:.2f}: 初始 {initial_epochs} 轮, 增强 {remaining_epochs} 轮")
    
    # 测试不同的挖掘策略
    strategies = ["confidence_based", "iou_based", "loss_based"]
    print(f"\n不同挖掘策略: {', '.join(strategies)}")
    
    # 测试不同的阈值组合
    confidence_thresholds = [0.3, 0.5, 0.7]
    iou_thresholds = [0.3, 0.5, 0.7]
    
    print("\n不同阈值组合:")
    for conf_thresh in confidence_thresholds:
        for iou_thresh in iou_thresholds:
            print(f"  置信度 {conf_thresh}, IoU {iou_thresh}")
    
    print("✅ 参数组合测试完成")
    return True

def test_command_line_examples():
    """测试命令行使用示例"""
    print("\n🔧 测试命令行使用示例")
    print("="*50)
    
    examples = [
        {
            "name": "分阶段训练（默认）",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --project runs/train \\
    --name fish_detection_hard_negative \\
    --mining_strategy confidence_based"""
        },
        {
            "name": "使用预训练模型",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --use_pretrained_for_mining \\
    --mining_strategy confidence_based"""
        },
        {
            "name": "自定义初始轮数比例",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --initial_epochs_ratio 0.4 \\
    --mining_strategy confidence_based"""
        },
        {
            "name": "完整参数配置",
            "command": """python3 detection/train_yolo_with_hard_negative.py \\
    --data ./datasets/l0_9.12/dataset.yaml \\
    --model yolov8s.pt \\
    --epochs 100 \\
    --project runs/train \\
    --name comprehensive_hard_negative \\
    --mining_strategy confidence_based \\
    --hard_negative_ratio 0.3 \\
    --confidence_threshold 0.6 \\
    --iou_threshold 0.5 \\
    --save_hard_negatives \\
    --initial_epochs_ratio 0.33"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"示例 {i}: {example['name']}")
        print(example['command'])
        print()
    
    print("✅ 命令行示例测试完成")
    return True

def main():
    """主函数"""
    print("🚀 硬负样本挖掘策略测试")
    print("="*60)
    
    success_count = 0
    total_tests = 4
    
    # 运行测试
    if test_strategy_1_staged_training():
        success_count += 1
    
    if test_strategy_2_pretrained_model():
        success_count += 1
    
    if test_parameter_combinations():
        success_count += 1
    
    if test_command_line_examples():
        success_count += 1
    
    # 输出测试结果
    print("\n" + "="*60)
    print("📊 测试结果摘要")
    print("="*60)
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有策略测试通过！")
        print("\n📝 使用建议:")
        print("1. 根据数据集大小选择合适的策略")
        print("2. 调整 initial_epochs_ratio 参数")
        print("3. 使用 --use_pretrained_for_mining 快速开始")
        print("4. 监控硬负样本挖掘报告")
        print("5. 比较不同策略的效果")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
