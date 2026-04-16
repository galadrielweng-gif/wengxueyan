# TASK1 - Medical Entity Extraction

## 任务说明
本任务基于中文病例PDF，使用DeepSeek大模型进行医学实体抽取。

## 方法流程
1. 读取PDF病例文本（PyMuPDF）
2. 构建医学信息抽取Prompt
3. 调用DeepSeek API
4. 输出结构化JSON结果

## 输出字段
- patient_info
- symptoms
- medical_history
- diagnosis
- treatment

## 运行方法
python extract_entities.py

## 输出文件
task1_result.json