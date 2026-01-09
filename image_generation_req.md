# 📊 项目绘图需求说明书 (Data Viz Requirements)

## 1. 概览
本文档旨在说明如何使用配套的 `data.txt` 生成项目汇报所需的 6 张关键图表。
**数据源**：所有绘图数据均已标准化存储在 `data.txt` 中，请编写 Python 脚本 (推荐使用 Matplotlib/Seaborn) 读取并绘制。

## 2. 通用技术要求
* **工具库**：`matplotlib`, `numpy`, `networkx` (可选，用于UML)
* **字体支持**：图中包含大量中文，请确保脚本开头配置了中文字体（如 SimHei 或 Microsoft YaHei）。
    ```python
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ```
* **风格**：学术/论文风格，背景简洁（白色），高分辨率（DPI >= 300）。

---

## 3. 图表详细说明

### [图1] RAG 核心原理流程图
* **数据标识**：`=== FIGURE 1: RAG PIPELINE ===`
* **尺寸建议**：14cm x 8cm
* **绘图思路**：
    * 这是一个**概念+文本**的组合图。
    * 上半部分绘制流程：Query -> Retrieval (Milvus) -> Prompt -> LLM -> Answer。
    * 下半部分展示文本对比框：左侧放 `No_RAG_Response` (体现幻觉/重复)，右侧放 `RAG_Response` (体现准确性)。
    * **重点**：高亮右侧答案中的医学关键词（如“三多一少”）。

### [图2] 模型效能对比雷达图
* **数据标识**：`=== FIGURE 2: MODEL RADAR CHART ===`
* **尺寸建议**：10cm x 10cm
* **绘图思路**：
    * 使用标准的雷达图（Spider Plot）。
    * **维度**：`Instruction Following`, `Medical Accuracy`, `Logical Consistency`, `Response Speed`。
    * **对比**：
        * GPT-2 (Baseline)：使用灰色或浅蓝色虚线填充，面积较小。
        * Qwen2.5 (Ours)：使用红色或深蓝色实线填充，面积较大，覆盖前者。
    * **注意**：`Response Speed` 分数越高代表越好（已在数据中处理好，直接画即可）。

### [图3] L2 与余弦相似度检索分布对比
* **数据标识**：`=== FIGURE 3: METRIC DISTRIBUTION ===`
* **尺寸建议**：12cm x 7cm
* **绘图思路**：
    * **图表类型**：双子图 (Subplots) 或 双折线图。
    * X轴：Index (数据点序号)
    * Y轴：Score (分数)
    * **展示逻辑**：
        * **L2 (欧氏距离)**：数据点应该比较散乱，表示在不做归一化时距离分布方差大。
        * **Cosine (余弦相似度)**：数据应该相对集中或有明显的梯度（如果排序的话），体现其在高维语义检索中的稳定性。
    * *建议*：可以将 Cosine 数据在绘图前先 `sort` 一下，画成平滑曲线，对比 L2 的震荡，视觉效果更好。

### [图4] 语义分块与重叠示意图
* **数据标识**：`=== FIGURE 4: CHUNKING SCHEMATIC ===`
* **尺寸建议**：14cm x 5cm
* **绘图思路**：
    * **可视化形式**：水平条形图 (Horizontal Bars) 或 文本块堆叠。
    * **布局**：
        * 第一行：完整的 `Original_Text` 长条。
        * 下放：交错排列的 `Chunk_1`, `Chunk_2`...
    * **关键点**：用半透明颜色标记出 **Overlap (重叠区)**，并拉出箭头注释“Overlap 保留完整实体”。
    * 使用数据中的 `[Start:End]` 坐标来确定色块长度。

### [图5] 系统组件依赖关系图 (UML)
* **数据标识**：`=== FIGURE 5: SYSTEM ARCHITECTURE ===`
* **尺寸建议**：12cm x 10cm
* **绘图思路**：
    * 推荐使用 `networkx` 库或手动画框。
    * **节点**：`RAGService`, `VectorStoreService` 等类名。
    * **连线**：有向箭头，表示 `Depends_On` 关系。
    * **布局**：将 `RAGService` 放在最上层或中心，体现它是“Facade (门面)”或“Controller (控制器)”，其他 Service 位于底层。

### [图6] 系统性能监控统计图
* **数据标识**：`=== FIGURE 6: PERFORMANCE STATS ===`
* **尺寸建议**：12cm x 8cm
* **绘图思路**：
    * **双Y轴图表 (Dual Axis)**。
    * **X轴**：文档数量 (`Doc_Count`: 100 -> 2000)。
    * **左Y轴 (柱状图)**：`Search_Time_ms` (检索耗时)。颜色建议：蓝色。
    * **右Y轴 (折线图)**：`Generate_Time_ms` (生成耗时)。颜色建议：橙色。
    * **结论暗示**：检索耗时随数据量线性/对数增长（但依然很快），生成耗时相对稳定（只受 Prompt 长度影响）。

---