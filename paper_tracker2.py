import os
import arxiv
import fitz  # PyMuPDF，用于PDF文本提取
from datetime import datetime
from openai import OpenAI
import os

# 临时环境变量配置
os.environ["ARK_API_KEY"] = "cc9c6d90-0cb9-4aa3-a991-b54149b68814" # 1
# 通过 OpenAI SDK 初始化客户端，注意替换 base_url 和模型 endpoint ID
client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本，并返回文本字符串
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"提取 {pdf_path} 文本失败: {e}")
    return text

def summarize_paper(text):
    """
    调用 deepseek API 对论文文本生成摘要
    采用 non-streaming 模式，返回摘要字符串
    """
    prompt = f"现有一篇最新论文，请你深入阅读并进行全面分析，主要从以下几个角度展开讨论：\
        1. **核心创新点**：总结论文提出的主要算法或技术创新，解释其原理和与现有方法的差异。\
        2. **实验与 验证**：评估论文中的实验设计、数据集、评测指标和结果，讨论这些结果的说服力和可靠性。\
        3. **应用场景**：重点探讨论文中的技术如何在搜索引擎业务中发挥作用。请分析该技术在改进排序、召回、用户体验、数据处理等环节上的潜力与优势。\
        4. **挑战与改进**：指出在将该技术落地到实际搜索引擎系统时可能遇到的技术挑战和工程难题，并提出可能的改进方案或替代思路。\
        5. **未来展望**：结合当前搜索引擎的发展趋势，讨论该论文的技术对未来产品演进的启示和影响。\
        请根据上述要求，生成一个结构清晰、逻辑严谨且具有深度的总结报告，既包括对论文内容的客观描述，也包含你对其在搜索引擎业务中应用价值的专业见解和建议。\
        \n{text}"
    
    # 构造对话消息，这里设定系统身份和用户输入
    messages = [
        {"role": "system", "content": "你是一位资深的搜索算法工程师，请你从专业角度阅读这篇论文，生成一个包含核心创新、实验设计和在搜索引擎业务中应用价值的详细摘要。"},
        {"role": "user", "content": prompt}
    ]
    
    # 调用 API 生成摘要（使用 Non-streaming 模式）
    completion = client.chat.completions.create(
        model="ep-20250213001139-x7xn6", 
        messages=messages,
    )
    
    summary = completion.choices[0].message.content
    return summary

def process_all_pdfs_and_summarize(folder_path):
    """
    遍历指定文件夹下所有PDF文件，提取文本并调用 deepseek API 生成摘要，
    最后将每篇论文的摘要打印出来
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"\n正在处理文件: {filename}")
            
            # 提取 PDF 文本
            text = extract_text_from_pdf(pdf_path)
            if not text:
                print("未能提取到文本，跳过该文件。")
                continue
            
            print("正在调用 API 生成摘要...")
            summary = summarize_paper(text)
            if summary:
                print("生成的摘要：")
                print(summary)
            else:
                print("摘要生成失败。")


def search_and_download(keywords, max_results=5, download_dir='downloads'):
    query = " AND ".join(keywords)  # OR/AND 参数传入
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    for result in search.results():
        print(f"正在下载: {result.title}")
        try:
            # 自动下载 PDF 到指定目录（该方法会以论文 arXiv ID 作为文件名保存）
            result.download_pdf(dirpath=download_dir)
            print("下载完成！")
        except Exception as e:
            print(f"下载失败: {result.title}，错误信息：{e}")


if __name__ == '__main__':
    # 以当前日期（格式为 YYYYMMDD）作为文件夹名称，例如 "./20250213"
    today = datetime.today().strftime('%Y%m%d')
    folder_path = f"./{today}"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    keywords = ["Information Retrieval", "Search"]
    search_and_download(keywords=keywords, max_results=1, download_dir=folder_path)
    # 开始处理指定文件夹中的所有 PDF 文件
    process_all_pdfs_and_summarize(folder_path)
