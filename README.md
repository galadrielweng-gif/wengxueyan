# 医疗大模型新人三阶段挑战项目

本项目为医疗大模型研究组新人训练任务，包含三个阶段（TASK1–TASK3），覆盖大模型API调用、机器学习建模与AI辅助科研写作。

---

# 项目整体结构

```text
.
├── TASK1/
│   ├── *.py                 # Python源码（API调用与信息抽取）
│   ├── *.json               # 模型输出结果
│   ├── *.pdf                # 原始病例文件
│   ├── requirements.txt     # 依赖说明
│
├── TASK2/
│   ├── *.ipynb / *.py       # 数据分析与建模代码
│   ├── heart_failure_*.csv  # 原始数据集
│   ├── figures/             # 图表（ROC、热力图、特征重要性等）
│   ├── latex/               # LaTeX源码
│   │   ├── main.tex
│   │   ├── references.bib
│   ├── paper.pdf            # 编译后的论文
│   ├── requirements.txt
│
├── TASK3/
│   ├── guide.md             # 医疗大模型新人快速上手指南
│   ├── guide.pdf            # 导出PDF版本
│
├── AI_Chat_Records.md       # AI协作对话记录（全部任务汇总）
└── README.md
