# 码头护轮坎人员闯入检测系统


---

## 核心方案
- **异构模型集成 (Ensemble)**：同步聚合 YOLO11m, YOLO13l, RT-DETR v2, DEIM 以及 RT-DETRv4 等不同家族的检测模型。
- **并集决策 (OR Strategy)**：基于“非对称代价”考量，只要任一模型触发报警，系统即输出预警，最大化 **召回率 (Recall)**。
- **逻辑演进**：支持从基础点判定到带 10px 缓冲区的 IoA 区域判定（Logic C），增强了对检测框抖动的鲁棒性。

---

## 性能表现 (测试平台: NVIDIA RTX 4090)

| 判定逻辑 | 方案 | 召回率 (Recall) | 精确率 (Precision) | 推理耗时 (Speed) |
| :--- | :--- | :--- | :--- | :--- |
| **Logic C** | **五模型集成 (Ensemble)** | **0.9793** | 0.9753 | ~200ms |

> **结论分析**：在码头安防等生命安全敏感场景下，漏报代价远高于误报。集成方案通过牺牲部分推理时延，实现了对风险样本的最严密覆盖。

---

## 运行

### 1. 创建虚拟环境
```bash
conda env create -f environment.yaml
conda activate hulunkan_env
```

### 2. 运行测试
执行主脚本进行批量图像推理与指标统计：
```bash
python process.py
```