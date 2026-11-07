# PhyTCR-Net 主动脉压力与流量波形数据集

## 下载完整数据集

- [v1.0.1完整数据ZIP](https://github.com/2shelter22/PhyTCR-Net-Aortic-Hemodynamics-Data/releases/download/v1.0.1/PhyTCR_Net_Aortic_Hemodynamics_Data_v1.0.1.zip)
- [版本发布页](https://github.com/2shelter22/PhyTCR-Net-Aortic-Hemodynamics-Data/releases/tag/v1.0.1)
- 压缩包62,075,840字节，解压后1,573个文件。
- ZIP SHA256：`ebddc35055bf5dead5ad425ccfc7720a54b0a523279ee45a245697f78265a581`。

本Git仓库提供轻量说明、划分、来源和复算接口，完整数值数据及预测数组保存在版本附件中。请先下载并解压ZIP，再在**解压后的数据包根目录**运行下述命令，不要在轻量Git检出目录直接运行完整数据核验。ZIP内含完整文件哈希清单。

### 数据集简介

本数据集收录70例主动脉几何模型的截面压力与体积流量波形，并提供有向血管树、入口与出口边界条件、降阶物理模型结果以及预测评价材料。根据原始几何数据的来源关系，70例模型划分为63个来源组，用于来源分组五折交叉验证。数据可用于血流动力学降阶建模、血管拓扑相关学习与完整心动周期波形预测研究。

CFD监督标签包含每个心动周期的64个时间采样点，RI–RCR计算结果提供相应的物理基础响应。数据包还包含PhyTCR-Net及直接神经网络对照模型的外层测试预测结果、数据划分和主比较评分程序。

### 文件组成

| 路径 | 内容 |
|---|---|
| `data/cases/` | 逐例血管图、边界条件、CFD截面标签、采样截面、时相信息、RI–RCR计算结果与输入特征。 |
| `data/cases.json` | 案例标识、来源组标识及基本统计。 |
| `splits/five_fold.json` | 五折的训练、内层验证与外层测试案例及随机种子。 |
| `results/` | 主比较的外层测试压力与流量预测及评价汇总。 |
| `code/`、`scripts/` | 降阶模型与神经网络定义、数据读取和评价工具。 |
| `DATA_DICTIONARY.md` | 数组、形状、计量单位、有效截面掩码及特征定义。 |
| `SOURCES.md`、`licenses/` | 数据来源、推荐引用和上游许可。 |

### 快速开始

从数据集根目录运行以下命令，核对数据结构并复算主要评分：

```bash
python -B scripts/verify_data.py
python -B scripts/replay_scores.py
```

读取单例数据：

```bash
python -B examples/load_case.py --case aorta_017
```

使用 RI–RCR 计算复算脚本时，在所用Python环境安装SciPy后执行：

```bash
python -B scripts/replay_rom.py
```

### 数据与单位

`labels.npz/pressure_node` 的单位为 dyn/cm²，除以1333.22后得到mmHg；`labels.npz/flow_edge_cm3_s` 的单位为cm³/s，与mL/s数值相同。节点与边的布尔掩码分别标识用于评价的有效采样截面。每例 CFD 标签包含64个时间采样点，其时间坐标由 `phases.npz` 给出。有向血管树以入口为根，边的起点、终点索引从零开始。字段、单位和数组形状详见 `DATA_DICTIONARY.md`。

### 评价与使用

数据集按照几何来源组织五折交叉验证，学习模型采用三个随机种子进行评价。发布的主要比较预测共420份，涵盖PhyTCR-Net与直接神经网络对照模型。评分按有效截面计算压力和流量的RMSE，分别以mmHg与mL/s表示。数据包同时提供分组汇总方法、模型配置与可运行的评价脚本。

### 数据来源与引用

血管几何来源于 Vascular Model Repository（VMR）。来源论文、几何来源标识及原始权利声明见 `SOURCES.md` 和 `licenses/VMR/`。使用本数据集时，请按实际复用内容引用VMR来源论文及相应的PhyTCR-Net研究论文或正式数据集记录。

### 许可与署名

研究原创数据和说明文档采用 **CC BY-NC 4.0（署名—非商业性使用）**；原创代码采用 **PhyTCR非商业署名软件许可1.0**。仅允许非商业使用，必须保留署名、来源标注和许可声明，修改后分享须注明改动。VMR源材料继续遵守上游原许可。

各组件条款见 [LICENSES.md](LICENSES.md)，署名信息见 [ATTRIBUTION.md](ATTRIBUTION.md)，逐例官方来源标识和链接见 [data/provenance_index.json](data/provenance_index.json)。
