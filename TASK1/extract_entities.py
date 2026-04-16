import json
import fitz  # PyMuPDF
from openai import OpenAI

# =========================
# 1. DeepSeek API 配置
# =========================
client = OpenAI(
    api_key="sk-7f80493de8f04ad7ab59ed4fdc73308c",
    base_url="https://api.deepseek.com"
)

# =========================
# 2. 读取PDF文本
# =========================
pdf_path = r"E:\科研组\测试\wengxueyan-main\TASK1\A case of portal vein recanalization and symptomatic heart failure.pdf"

doc = fitz.open(pdf_path)

text = ""
for page in doc:
    text += page.get_text()

# 如果PDF太长，截断（防止超token）
text = text[:6000]

# =========================
# 3. Prompt设计（核心得分点）
# =========================
prompt = f"""
你是一名医学信息抽取助手，请从以下病例文本中提取结构化信息，并严格输出JSON：

字段：
- patient_info（患者基本信息）
- symptoms（主要症状）
- medical_history（既往史）
- diagnosis（诊断结果）
- treatment（治疗方案）

要求：
1. 必须输出合法JSON
2. 不要任何解释
3. 没有信息填 null

病例内容：
{text}
"""

# =========================
# 4. 调用 DeepSeek
# =========================
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

result_text = response.choices[0].message.content

# =========================
# 5. 保存 JSON
# =========================
clean_text = re.sub(r"```json|```", "", result_text).strip()

try:
    result_json = json.loads(clean_text)
except:
    result_json = {"raw_output": result_text}

output_path = r"E:\科研组\测试\wengxueyan-main\TASK1\task1_result.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=4)

print("TASK1完成 ✔ 结果已保存：", output_path)
