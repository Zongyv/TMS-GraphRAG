import os
import logging  # 导入日志模块
import time  # 导入时间模块
from pathlib import Path  # 导入路径处理模块
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem  # 导入文档处理相关类型
from docling.datamodel.base_models import FigureElement, InputFormat, Table  # 导入数据模型基类
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice, \
    RapidOcrOptions, EasyOcrOptions  # 导入PDF处理管道选项
from docling.document_converter import DocumentConverter, PdfFormatOption  # 导入文档转换器
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
)
import fitz  # PyMuPDF
import re
import pymupdf4llm  # 备用方案
import pdfplumber  # 备用方案
from pypdf import PdfReader  # 备用方案

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

IMAGE_RESOLUTION_SCALE = 5.0  # 图片分辨率缩放比例

logging.basicConfig(level=logging.INFO)  # 配置日志级别为INFO


def extract_doi_from_headers_footers(pdf_path):
    """从PDF的页眉页脚中提取DOI"""
    doi_patterns = [
        r'DOI:\s*([10]\.\d+/[^\s\n]+)',
        r'doi:\s*([10]\.\d+/[^\s\n]+)',
        r'https?://doi\.org/([10]\.\d+/[^\s\n]+)',
        r'https?://dx\.doi\.org/([10]\.\d+/[^\s\n]+)',
        r'\b(10\.\d+/[^\s\n]+)\b'
    ]
    
    def clean_doi(doi):
        """清理DOI，检测分隔符后是否跟了大段文本"""
        print(f"原始DOI: {doi}")
        
        # 移除空格
        cleaned = re.sub(r'\s+', '', doi)
        print(f"移除空格后: {cleaned}")
        
        # 首先检查并移除ISSN号模式 (XXXX-XXXX格式)
        issn_pattern = r'(\d{4}-\d{3}[\dX]).*$'
        issn_match = re.search(issn_pattern, cleaned)
        if issn_match:
            issn_start = issn_match.start(1)
            if issn_start > 0:  # 确保ISSN不是在开头
                cleaned = cleaned[:issn_start]
                print(f"移除ISSN后: {cleaned}")
        
        # 检查DOI后面是否跟了常见的无关词汇
        common_suffixes = [
            'Received', 'Accepted', 'Published', 'Available', 'online',
            'Introduction', 'Abstract', 'Background', 'Methods', 'Results',
            'Discussion', 'Conclusion', 'References', 'Acknowledgments',
            'depression', 'treatment', 'study', 'patients', 'clinical',
            'therapy', 'efficacy', 'randomized', 'controlled', 'trial',
            'Journal', 'pISSN', 'BRAIN', 'Repetitive', 'ARTICLETHEMATIC',
            'E-mail', 'researchopen','copyright','www','NIH','maintained',
            'IOS','Research','Author','Correspondence'
        ]
        
        # 检查是否以这些词汇结尾（不区分大小写）
        for suffix in common_suffixes:
            pattern = re.compile(f'(.+?)({re.escape(suffix)}.*)$', re.IGNORECASE)
            match = pattern.search(cleaned)
            if match:
                potential_doi = match.group(1)
                # 验证截断后的部分是否是有效DOI
                if re.match(r'^10\.\d+/', potential_doi):
                    cleaned = potential_doi
                    print(f"移除后缀 '{suffix}' 后: {cleaned}")
                    break
        
        # DOI中常见的分隔符
        separators = ['.', '-', '_', '/']
        
        # 从右到左扫描，寻找可能的截断点
        for i in range(len(cleaned) - 1, -1, -1):
            char = cleaned[i]
            if char in separators and i < len(cleaned) - 1:
                # 获取分隔符后面的部分
                remaining = cleaned[i+1:]
                print(f"分隔符 '{char}' 在位置 {i}，后面内容: {remaining}")
                
                # 检查后面是否有连续的长文本（可能是文章内容）
                if len(remaining) > 15:
                    print(f"后面内容长度 {len(remaining)} > 15")
                    
                    # 检查是否包含明显的文章关键词
                    article_keywords = [
                        'introduction', 'abstract', 'background', 'methods', 
                        'results', 'discussion', 'conclusion', 'depression',
                        'treatment', 'study', 'patients', 'clinical', 'original',
                        'jcn', 'neurological'
                    ]
                    
                    remaining_lower = remaining.lower()
                    keyword_found = any(keyword in remaining_lower for keyword in article_keywords)
                    print(f"包含文章关键词: {keyword_found}")
                    
                    if keyword_found:
                        # 截断到当前分隔符
                        cleaned = cleaned[:i]
                        print(f"根据关键词截断为: {cleaned}")
                        break
                    
                    # 或者检查是否有连续的大写字母开头的单词（如IntroductionDepression）
                    camel_case_pattern = re.search(r'[A-Z][a-z]+[A-Z][a-z]+', remaining)
                    print(f"包含驼峰模式: {bool(camel_case_pattern)}")
                    
                    if camel_case_pattern:
                        cleaned = cleaned[:i]
                        print(f"根据驼峰模式截断为: {cleaned}")
                        break
        
        # 移除末尾的标点符号，但保留DOI的有效字符
        cleaned = re.sub(r'[^\w\.\-/\(\)]+$', '', cleaned)
        print(f"最终清理后: {cleaned}")
        
        # 验证DOI格式
        if re.match(r'^10\.\d+/', cleaned):
            return cleaned
        return doi
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            rect = page.rect
            
            header_rect = fitz.Rect(0, 0, rect.width, rect.height * 0.15)
            header_text = page.get_text("text", clip=header_rect)
            
            footer_rect = fitz.Rect(0, rect.height * 0.85, rect.width, rect.height)
            footer_text = page.get_text("text", clip=footer_rect)
            
            for text in [header_text, footer_text]:
                # 先处理换行问题，将可能被分割的DOI连接起来
                text_cleaned = re.sub(r'\n(?=\w)', '', text)  # 移除单词中间的换行
                
                for pattern in doi_patterns:
                    match = re.search(pattern, text_cleaned, re.IGNORECASE)
                    if match:
                        doi = match.group(1).rstrip('.,;:')
                        doi = clean_doi(doi)
                        if re.match(r'^10\.\d+/', doi):
                            print(f"在第{page_num+1}页页眉/页脚中找到DOI: {doi}")
                            doc.close()
                            return doi
        
        doc.close()
        return None
    except Exception as e:
        print(f"提取页眉页脚DOI失败: {str(e)}")
        return None


def process_pdf_with_multiple_fallbacks(input_pdf_path, output_pdf_path):
    """使用多种备用方案处理PDF"""
    source = input_pdf_path
    doc_filename = Path(source).stem
    output_dir = Path(output_pdf_path, doc_filename)
    
    print(f"开始处理: {source}")
    
    # 先尝试从页眉页脚提取DOI
    header_footer_doi = extract_doi_from_headers_footers(source)
    
    # 方法1: 标准Docling处理
    try:
        print("尝试方法1: 标准Docling处理")
        return process_with_standard_docling(source, output_dir, header_footer_doi)
    except RuntimeError as e:
        if "Invalid code point" in str(e):
            print(f"方法1失败 (Invalid code point): {e}")
        else:
            print(f"方法1失败: {e}")
    except Exception as e:
        print(f"方法1失败: {e}")
    
    # 方法2: 关闭OCR的Docling处理
    try:
        print("尝试方法2: 关闭OCR的Docling处理")
        return process_with_docling_no_ocr(source, output_dir, header_footer_doi)
    except Exception as e:
        print(f"方法2失败: {e}")
    
    # 方法3: 使用PyMuPDF4LLM
    try:
        print("尝试方法3: PyMuPDF4LLM处理")
        return process_with_pymupdf4llm(source, output_dir, header_footer_doi)
    except Exception as e:
        print(f"方法3失败: {e}")
    
    # 方法4: 使用PyMuPDF基础处理
    try:
        print("尝试方法4: PyMuPDF基础处理")
        return process_with_pymupdf_basic(source, output_dir, header_footer_doi)
    except Exception as e:
        print(f"方法4失败: {e}")
    
    # 方法5: 使用pdfplumber
    try:
        print("尝试方法5: pdfplumber处理")
        return process_with_pdfplumber(source, output_dir, header_footer_doi)
    except Exception as e:
        print(f"方法5失败: {e}")
    
    print(f"所有方法都失败，跳过文件: {source}")
    return False

def process_with_standard_docling(source, output_dir, header_footer_doi):
    """标准Docling处理"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    conv_res = doc_converter.convert(source)
    return save_docling_results(conv_res, output_dir, header_footer_doi)

def process_with_docling_no_ocr(source, output_dir, header_footer_doi):
    """关闭OCR的Docling处理"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = False  # 关闭OCR

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    conv_res = doc_converter.convert(source)
    return save_docling_results(conv_res, output_dir, header_footer_doi)

def process_with_pymupdf4llm(source, output_dir, header_footer_doi):
    """使用PyMuPDF4LLM处理"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用pymupdf4llm提取markdown
    md_text = pymupdf4llm.to_markdown(source)
    
    # 保存markdown文件
    doc_filename = Path(source).stem
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    
    # 添加DOI信息
    if header_footer_doi:
        md_text = f"<!-- DOI: {header_footer_doi} -->\n\n{md_text}"
    
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    print(f"PyMuPDF4LLM处理完成: {md_filename}")
    return True

def process_with_pymupdf_basic(source, output_dir, header_footer_doi):
    """使用PyMuPDF基础处理"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(source)
    doc_filename = Path(source).stem
    
    # 提取文本
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n\n## Page {page_num + 1}\n\n{text}"
    
    # 提取图片
    image_counter = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha < 4:  # 确保不是CMYK
                    image_counter += 1
                    img_filename = output_dir / f"{doc_filename}-picture-{image_counter}.png"
                    pix.save(str(img_filename))
                    print(f"保存图片: {img_filename}")
                pix = None
            except Exception as e:
                print(f"提取图片失败: {e}")
    
    doc.close()
    
    # 保存markdown文件
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    
    # 添加DOI信息
    if header_footer_doi:
        full_text = f"<!-- DOI: {header_footer_doi} -->\n\n{full_text}"
    
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"PyMuPDF基础处理完成: {md_filename}")
    return True

def process_with_pdfplumber(source, output_dir, header_footer_doi):
    """使用pdfplumber处理"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc_filename = Path(source).stem
    full_text = ""
    
    with pdfplumber.open(source) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text:
                    full_text += f"\n\n## Page {page_num + 1}\n\n{text}"
                
                # 尝试提取表格
                tables = page.extract_tables()
                for table_num, table in enumerate(tables):
                    full_text += f"\n\n### Table {table_num + 1} (Page {page_num + 1})\n\n"
                    for row in table:
                        if row:
                            full_text += "| " + " | ".join([cell or "" for cell in row]) + " |\n"
            except Exception as e:
                print(f"处理第{page_num + 1}页时出错: {e}")
                continue
    
    # 保存markdown文件
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    
    # 添加DOI信息
    if header_footer_doi:
        full_text = f"<!-- DOI: {header_footer_doi} -->\n\n{full_text}"
    
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"pdfplumber处理完成: {md_filename}")
    return True

def save_docling_results(conv_res, output_dir, header_footer_doi):
    """保存Docling处理结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # 处理表格和图片
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            print("处理表格")
            table_counter += 1
            element_image_filename = (output_dir / f"{doc_filename}-table-{table_counter}.png")
            print(element_image_filename)
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            print("处理图片")
            picture_counter += 1
            element_image_filename = (output_dir / f"{doc_filename}-picture-{picture_counter}.png")
            print(element_image_filename)
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # 保存Markdown文件
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
    
    # 如果从页眉页脚找到了DOI，将其添加到Markdown文件开头
    if header_footer_doi:
        with open(md_filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查文件开头是否已经有DOI注释
        if not content.startswith('<!-- DOI:'):
            # 在文件开头添加DOI信息
            doi_header = f"<!-- DOI: {header_footer_doi} -->\n\n"
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(doi_header + content)
            
            print(f"已将DOI添加到Markdown文件开头: {header_footer_doi}")
        else:
            print(f"Markdown文件已包含DOI信息")
    else:
        print("未在页眉页脚中找到DOI")
    
    return True

def main():
    input_pdf_paths = "dataset/rTMS/pdf_processed"
    output_pdf_paths = "dataset/rTMS/markdown"
    if not os.path.exists(output_pdf_paths):
        os.makedirs(output_pdf_paths)
    
    skipped_files = []
    processed_files = []
    method_stats = {
        "方法1_标准Docling": 0,
        "方法2_无OCR_Docling": 0,
        "方法3_PyMuPDF4LLM": 0,
        "方法4_PyMuPDF基础": 0,
        "方法5_pdfplumber": 0,
        "全部失败": 0
    }
    
    for input_pdf_path in os.listdir(input_pdf_paths):
        if input_pdf_path.endswith(".pdf"):
            base_name = os.path.basename(input_pdf_path)
            base_name = ".".join(base_name.split(".")[:-1])
            if base_name in os.listdir(output_pdf_paths):
                print(f"skip {base_name}")
                continue
            
            print(f"\n{'='*50}")
            print(f"开始处理: {input_pdf_path}")
            start_time = time.time()
            
            try:
                success = process_pdf_with_multiple_fallbacks(
                    os.path.join(input_pdf_paths, input_pdf_path), 
                    output_pdf_paths
                )
                
                if success:
                    processed_files.append(input_pdf_path)
                    print(f"成功处理: {input_pdf_path}")
                else:
                    skipped_files.append(input_pdf_path)
                    method_stats["全部失败"] += 1
                    
            except Exception as e:
                print(f"处理失败: {input_pdf_path}, 错误: {e}")
                skipped_files.append(input_pdf_path)
                method_stats["全部失败"] += 1
            
            end_time = time.time()
            print(f"处理时间: {end_time - start_time:.2f} 秒")
    
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"成功处理: {len(processed_files)} 个文件")
    print(f"跳过文件: {len(skipped_files)} 个文件")
    
    print(f"\n处理方法统计:")
    for method, count in method_stats.items():
        if count > 0:
            print(f"  {method}: {count} 个文件")
    
    if skipped_files:
        print("\n跳过的文件列表:")
        for file in skipped_files:
            print(f"  - {file}")


if __name__ == "__main__":
    main()