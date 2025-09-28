# 机器人旋转指南

## 概述
本指南说明如何在XYZRPY格式下控制机器人基座旋转。

## XYZRPY格式说明
- **X, Y, Z**: 位置坐标 (毫米)
- **R (Roll)**: 绕X轴旋转 (弧度)
- **P (Pitch)**: 绕Y轴旋转 (弧度)  
- **Y (Yaw)**: 绕Z轴旋转 (弧度)

## 旋转角度转换
- 90度 = π/2 弧度 ≈ 1.57 弧度
- 180度 = π 弧度 ≈ 3.14 弧度
- 270度 = 3π/2 弧度 ≈ 4.71 弧度
- 360度 = 2π 弧度 ≈ 6.28 弧度

## 代码示例

### 1. 旋转基座90度 (Yaw轴)
```python
import math
rotation_angle = math.pi / 2  # 90度
ret = self.robot.linear_move([0, 0, 0, 0, 0, rotation_angle], 1, True, 400)
```

### 2. 旋转基座180度 (Yaw轴)
```python
import math
rotation_angle = math.pi  # 180度
ret = self.robot.linear_move([0, 0, 0, 0, 0, rotation_angle], 1, True, 400)
```

### 3. 旋转基座-90度 (反向旋转)
```python
import math
rotation_angle = -math.pi / 2  # -90度
ret = self.robot.linear_move([0, 0, 0, 0, 0, rotation_angle], 1, True, 400)
```

### 4. 其他轴旋转
```python
# Roll轴旋转 (绕X轴)
roll_angle = math.pi / 2
ret = self.robot.linear_move([0, 0, 0, roll_angle, 0, 0], 1, True, 400)

# Pitch轴旋转 (绕Y轴)
pitch_angle = math.pi / 2
ret = self.robot.linear_move([0, 0, 0, 0, pitch_angle, 0], 1, True, 400)
```

## 当前实现
在 `realtime_segmentation_3d.py` 中，机器人抓取后会执行以下动作序列：

1. 移动到目标位置
2. 向上移动250mm
3. 向Y+方向移动350mm
4. **旋转基座90度** (新增)
5. 关闭夹爪
6. 返回原始位置

## 注意事项
- 旋转角度使用弧度制
- 正值为顺时针旋转，负值为逆时针旋转
- 确保旋转不会导致机器人碰撞
- 建议在安全环境中测试旋转动作
- 如果旋转失败，机器人会返回原始位置并关闭夹爪

## 故障排除
如果旋转失败，检查：
1. 机器人是否在安全位置
2. 旋转角度是否合理
3. 机器人工作空间限制
4. 连接状态是否正常

