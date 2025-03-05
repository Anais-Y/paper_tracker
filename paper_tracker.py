import os
import arxiv
import fitz  # PyMuPDF，用于 PDF 文本提取
from datetime import datetime
from openai import OpenAI

# 初始化 OpenAI 客户端（用于调用 deepseek 模型）
client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 目标公司关键词列表（全部为英文名称）
TARGET_COMPANIES = [
    "Google", "Shopee", "Alibaba", "Qwen", "Xiaohongshu", "Meituan", "Tencent", "Baidu"
    "Facebook", "Amazon", "Microsoft", "Apple", "ByteDance", "Kuaishou", "jd"
]

def search_and_download(keywords, max_results=10, download_dir='downloads'):
    """
    根据指定关键词从 arXiv 上搜索论文，并自动下载 PDF 文件到 download_dir 目录。
    多个关键词之间以 " AND " 组合。
    """
    query = " AND ".join(keywords)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    # 将结果转换为列表，防止迭代器被提前消费
    results = list(search.results())
    print(results)
    
    for result in results:
        print(f"正在下载: {result.title}")
        try:
            result.download_pdf(dirpath=download_dir)
            print("下载完成！")
        except Exception as e:
            print(f"下载失败: {result.title}，错误信息：{e}")

def extract_text_from_pdf(pdf_path):
    """
    使用 PyMuPDF 从 PDF 文件中提取文本，返回文本字符串。
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"提取 {pdf_path} 文本失败: {e}")
    return text

def contains_target_company(text):
    """
    检查论文文本中是否包含目标公司关键词（不区分大小写）。
    返回 True 表示包含，False 表示不包含。
    """
    lower_text = text.lower()
    for company in TARGET_COMPANIES:
        if company.lower() in lower_text:
            return True
    return False

def summarize_paper(text):
    """
    调用 deepseek 模型生成摘要。
    为防止输入文本过长，这里只取文本前 5000 个字符。
    """
    prompt = (
        "现有一篇最新论文，请你深入阅读并进行全面分析，主要从以下几个角度展开讨论：\n"
        "1. **核心创新点**：总结论文提出的主要算法或技术创新，解释其原理和与现有方法的差异。\n"
        "2. **实验与验证**：评估论文中的实验设计、数据集、评测指标和结果，讨论这些结果的说服力和可靠性。\n"
        "3. **应用场景**：重点探讨论文中的技术如何在搜索引擎业务中发挥作用。请分析该技术在改进排序、召回、用户体验、数据处理等环节上的潜力与优势。\n"
        "4. **挑战与改进**：指出在将该技术落地到实际搜索引擎系统时可能遇到的技术挑战和工程难题，并提出可能的改进方案或替代思路。\n"
        "5. **未来展望**：结合当前搜索引擎的发展趋势，讨论该论文的技术对未来产品演进的启示和影响。\n"
        f"\n{text[:5000]}"
    )
    messages = [
        {"role": "system", "content": "你是一位资深的搜索算法工程师，请你从专业角度阅读论文，生成包含核心创新、实验设计和在搜索引擎业务中应用价值的详细摘要。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="ep-20250213001139-x7xn6",  # 替换为实际的模型 endpoint ID
            messages=messages,
        )
        summary = completion.choices[0].message.content
        return summary
    except Exception as e:
        print("调用 API 生成摘要失败：", e)
        return None

def process_all_pdfs_and_summarize(folder_path):
    """
    遍历 folder_path 下所有 PDF 文件，
    对每篇论文提取文本并检测是否包含目标大型互联网公司的信息，
    若满足条件则调用 deepseek 模型生成摘要，
    返回列表，每个元素为 (论文标题, 摘要) 的元组。
    这里默认以 PDF 文件名（去除后缀）作为论文标题。
    """
    outputs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            paper_title = os.path.splitext(filename)[0]
            print(f"\n正在处理文件: {filename}")
            
            text = extract_text_from_pdf(pdf_path)
            if not text:
                print("未能提取到文本，跳过该文件。")
                continue

            # 检查论文中是否包含目标公司的关键词
            if not contains_target_company(text):
                print(f"{paper_title} 未检测到目标公司信息，跳过。")
                continue

            print("检测到目标公司信息，正在调用 API 生成摘要...")
            summary = summarize_paper(text)
            if summary:
                print(f"生成的摘要（{paper_title}）：")
                print(summary)
                outputs.append((paper_title, summary))
            else:
                print(f"{paper_title} 摘要生成失败。")
    return outputs

def write_to_readme(date, outputs):
    """
    将当天所有论文的摘要写入 README.md 文件：
      - 第一行为当天日期（一级标题，例如 "# 20250213"）
      - 每篇论文的标题作为二级标题，摘要为正文内容
    如果 README.md 文件已存在，则追加当天的内容。
    """
    content = f"# {date}\n\n"
    for title, summary in outputs:
        content += f"## {title}\n\n"
        content += f"{summary}\n\n"
    
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    print("README.md 文件已更新。")

if __name__ == '__main__':
    # 使用当前日期（格式为 YYYYMMDD）作为文件夹名称，例如 "./20250213"
    today = datetime.today().strftime('%Y%m%d')
    folder_path = f"./{today}"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 已创建。")
    else:
        print(f"文件夹 {folder_path} 已存在。")
    
    # 提供多个搜索关键词，专注于搜索算法与信息检索领域
    keywords = [
        "search engine", "information retrieval", "ranking algorithms",
        # "query understanding", "relevance ranking", "user search behavior", "click-through rate prediction",
        # "deep learning retrieval"
    ]
    # 从 arXiv 搜索并下载论文到 folder_path
    search_and_download(keywords, max_results=10, download_dir=folder_path)
    
    # 处理下载到 folder_path 中的所有 PDF 文件，生成摘要（仅针对包含目标公司信息的论文）
    outputs = process_all_pdfs_and_summarize(folder_path)
    
    # 如果有摘要生成，则写入 README.md 文件
    if outputs:
        write_to_readme(today, outputs)
    else:
        print("没有生成任何摘要。")
