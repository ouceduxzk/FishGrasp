# 硬负样本挖掘 (Hard Negative Mining) 使用指南

## 概述

硬负样本挖掘是一种提高目标检测模型性能的重要技术。它通过识别模型预测错误的困难样本，并增加这些样本在训练中的权重，来提高模型对困难样本的识别能力。

## 主要功能

1. **自动识别困难样本**：识别高置信度但IoU低的假阳性样本
2. **动态调整权重**：为困难样本分配更高的训练权重
3. **统计分析**：提供详细的困难样本分布分析
4. **可视化**：生成困难样本分布的可视化图表
5. **报告生成**：自动生成挖掘报告和性能指标

## 文件结构

```
detection/
├── train_yolo_with_hard_negative.py    # 完整的训练脚本（支持硬负样本挖掘）
├── hard_negative_mining.py             # 核心挖掘工具
├── example_hard_negative_usage.py      # 使用示例
└── README_hard_negative_mining.md      # 本文档
```

## 快速开始

### 1. 基本使用

```python
from detection.hard_negative_mining import HardNegativeMiner

# 创建挖掘器
miner = HardNegativeMiner(
    confidence_threshold=0.5,  # 置信度阈值
    iou_threshold=0.5,         # IoU阈值
    hard_negative_ratio=0.3,   # 硬负样本比例
    save_samples=True          # 保存困难样本
)

# 进行硬负样本挖掘
hard_negatives = miner.find_hard_negatives(predictions, ground_truth)

# 生成报告
report = miner.generate_report()
```

### 2. 完整训练流程

```bash
# 使用支持硬负样本挖掘的训练脚本
python3 detection/train_yolo_with_hard_negative.py \
    --data ./datasets/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --mining_strategy confidence_based \
    --hard_negative_ratio 0.3 \
    --confidence_threshold 0.5 \
    --save_hard_negatives
```

### 3. 运行示例

```bash
# 运行使用示例
python3 detection/example_hard_negative_usage.py
```

## 参数说明

### HardNegativeMiner 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `confidence_threshold` | float | 0.5 | 置信度阈值，高于此值的预测被认为是高置信度 |
| `iou_threshold` | float | 0.5 | IoU阈值，用于判断预测与真实标注的匹配程度 |
| `hard_negative_ratio` | float | 0.3 | 硬负样本比例，用于控制挖掘的样本数量 |
| `save_samples` | bool | False | 是否保存困难样本的图像和标注 |
| `output_dir` | str | "hard_negatives" | 困难样本输出目录 |

### 挖掘策略

1. **confidence_based**：基于置信度的挖掘
   - 识别高置信度但IoU低的假阳性样本
   - 适合处理模型过度自信的情况

2. **iou_based**：基于IoU的挖掘
   - 识别IoU在阈值附近的困难样本
   - 适合处理边界模糊的样本

3. **loss_based**：基于损失的挖掘
   - 根据预测损失识别困难样本
   - 需要访问模型的损失信息

## 输出结果

### 1. 硬负样本信息

每个硬负样本包含以下信息：
```python
{
    'type': 'false_positive',           # 样本类型
    'prediction': {...},                # 预测结果
    'ground_truth': {...},              # 真实标注
    'iou': 0.3,                        # IoU值
    'confidence': 0.8,                  # 置信度
    'image_path': 'path/to/image.jpg'   # 图像路径
}
```

### 2. 统计报告

```python
{
    'statistics': {
        'total_predictions': 1000,
        'hard_negatives': 150,
        'false_positives': 120,
        'false_negatives': 30,
        'true_positives': 800
    },
    'metrics': {
        'precision': 0.87,
        'recall': 0.96,
        'f1_score': 0.91,
        'hard_negative_rate': 0.15
    }
}
```

### 3. 可视化图表

- 置信度 vs IoU 散点图
- 置信度分布直方图
- IoU分布直方图
- 困难样本类型分布

## 集成到现有训练流程

### 1. 在训练循环中集成

```python
# 在验证阶段进行硬负样本挖掘
def validate_with_hard_negative_mining(model, val_loader, miner):
    model.eval()
    all_hard_negatives = []
    
    for batch_idx, (images, targets) in enumerate(val_loader):
        with torch.no_grad():
            predictions = model(images)
        
        # 转换为挖掘器需要的格式
        preds = convert_predictions(predictions)
        gts = convert_targets(targets)
        
        # 进行硬负样本挖掘
        hard_negatives = miner.find_hard_negatives(preds, gts)
        all_hard_negatives.extend(hard_negatives)
    
    return all_hard_negatives
```

### 2. 调整训练策略

```python
# 根据硬负样本调整训练
def adjust_training_with_hard_negatives(hard_negatives, train_loader):
    # 增加困难样本的权重
    sample_weights = calculate_sample_weights(hard_negatives)
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(sample_weights, len(train_loader.dataset))
    
    # 使用加权采样器
    weighted_loader = DataLoader(train_loader.dataset, sampler=sampler)
    
    return weighted_loader
```

## 最佳实践

### 1. 参数调优

- **置信度阈值**：根据模型的实际表现调整，通常设置为0.5-0.7
- **IoU阈值**：根据任务要求调整，严格任务使用0.7，宽松任务使用0.5
- **硬负样本比例**：开始时使用0.2-0.3，根据效果调整

### 2. 训练策略

- 在训练初期进行硬负样本挖掘
- 定期（每10-20个epoch）重新挖掘
- 使用挖掘结果调整数据增强策略
- 考虑使用困难样本进行额外训练

### 3. 性能监控

- 监控硬负样本率的变化
- 观察精确率和召回率的变化
- 分析困难样本的分布特征
- 根据挖掘结果调整模型架构

## 常见问题

### Q1: 硬负样本挖掘会增加训练时间吗？

A: 是的，硬负样本挖掘会增加一些计算开销，但通常不会显著影响训练时间。可以通过调整挖掘频率来平衡性能和效率。

### Q2: 如何选择合适的置信度阈值？

A: 建议从0.5开始，然后根据模型的实际表现进行调整。如果模型过度自信，可以降低阈值；如果模型过于保守，可以提高阈值。

### Q3: 硬负样本挖掘适用于所有数据集吗？

A: 硬负样本挖掘特别适用于包含大量困难样本的数据集。对于简单数据集，效果可能不明显。

### Q4: 如何处理类别不平衡的问题？

A: 可以为不同类别设置不同的阈值，或者使用类别特定的硬负样本挖掘策略。

## 扩展功能

### 1. 自定义挖掘策略

```python
class CustomMiner(HardNegativeMiner):
    def find_hard_negatives(self, predictions, ground_truth, image_path=None):
        # 实现自定义的挖掘逻辑
        pass
```

### 2. 集成其他技术

- 与Focal Loss结合使用
- 与数据增强技术结合
- 与模型蒸馏结合

### 3. 实时挖掘

```python
# 在推理过程中进行实时挖掘
def real_time_mining(model, image):
    prediction = model(image)
    hard_negatives = miner.find_hard_negatives(prediction, ground_truth)
    return prediction, hard_negatives
```

## 总结

硬负样本挖掘是提高目标检测模型性能的有效技术。通过合理使用本工具，可以：

1. 识别模型预测的困难样本
2. 提高模型对困难样本的识别能力
3. 改善模型的整体性能
4. 获得详细的性能分析报告

建议在实际项目中逐步集成硬负样本挖掘功能，并根据具体需求调整参数和策略。

