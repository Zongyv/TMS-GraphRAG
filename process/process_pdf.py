import pdfplumber
from pypdf import PdfReader, PdfWriter

import os
def detect_table_orientation(page):
    """
    检测页面中表格的方向。
    返回 True 如果页面包含竖向表格；False 如果页面包含横向表格或没有表格。
    """
    # 提取页面中的线条
    lines = page.lines
    if not lines:
        return False  # 没有检测到表格

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        # 根据线条宽度和高度比例判断方向
        if abs(line['width']) > abs(line['height']):
            horizontal_lines.append(abs(line['width']))
        else:
            vertical_lines.append(abs(line['height']))

    # 统计水平线和垂直线的总长度
    total_horizontal_length = sum(horizontal_lines) if horizontal_lines else 0
    total_vertical_length = sum(vertical_lines) if vertical_lines else 0

    # 调试信息：打印总长度
    print(f"Page {page.page_number}: Horizontal lines length: {total_horizontal_length}, "
          f"Vertical lines length: {total_vertical_length}")

    # 判断表格方向
    # 如果垂直线总长度大于水平线总长度的1.3倍 且 垂直线数量多于水平线数量，则认为是竖向表格
    is_vertical_by_lines = (total_vertical_length > 1.3 * total_horizontal_length) and (len(vertical_lines) > len(horizontal_lines))

    is_vertical_by_text = False
    if not is_vertical_by_lines:
        is_vertical_by_text = check_chinese_serial_numbers(page)

    return is_vertical_by_lines or is_vertical_by_text

def check_chinese_serial_numbers(page):
    """
    检查页面中是否存在竖排的汉字序号
    返回：True 如果存在竖排序号；False 否则。
    """

    # 定义常见的汉字序号
    chinese_serial_numbers = ["号序"]

    # 提取页面中的文本块
    text_blocks = page.extract_words()

    # 遍历文本块，检查是否存在竖排的序号
    for block in text_blocks:
        text = block["text"]
        print(text)
        if text in chinese_serial_numbers:
            # 计算文本块的宽高比
            width = block["x1"] - block["x0"]
            height = block["bottom"] - block["top"]
            print(width, height)
            if height > width:  # 如果高度大于宽度，则认为是竖排
                print(f"Page {page.page_number} 检测到竖排序号：'{text}'")
                return True



def process_pdf(input_pdf_path, output_pdf_path):
    """
    处理 PDF 文件，将包含竖向表格的页面设置为横向，并输出新的 PDF。
    """
    print(f"开始处理PDF: {input_pdf_path}")
    
    # 检查文件基本信息
    if not os.path.exists(input_pdf_path):
        print(f"文件不存在: {input_pdf_path}")
        return
        
    if os.path.getsize(input_pdf_path) == 0:
        print(f"文件为空: {input_pdf_path}")
        return
    
    try:
        # 尝试用pypdf读取
        try:
            reader = PdfReader(input_pdf_path)
            print(f"pypdf读取成功: {len(reader.pages)} 页")
        except Exception as pypdf_error:
            print(f"pypdf读取失败: {pypdf_error}")
            print(f"错误类型: {type(pypdf_error).__name__}")
            print("pypdf无法读取文件，直接复制原文件")
            import shutil
            shutil.copy2(input_pdf_path, output_pdf_path)
            print(f"已复制原文件到: {output_pdf_path}")
            return
        
        # 检查是否加密
        if reader.is_encrypted:
            print("PDF文件已加密，尝试解密...")
            try:
                reader.decrypt("")
                print("解密成功")
            except:
                print("解密失败，跳过处理")
                import shutil
                shutil.copy2(input_pdf_path, output_pdf_path)
                return
        
        # 用pdfplumber处理
        with pdfplumber.open(input_pdf_path) as pdf:
            print(f"pdfplumber读取结果: {len(pdf.pages)} 页")
            
            if len(pdf.pages) == 0:
                print("pdfplumber无法读取页面，直接复制原文件")
                import shutil
                shutil.copy2(input_pdf_path, output_pdf_path)
                return
            
            writer = PdfWriter()
            successful_pages = 0
            failed_pages = 0

            for i, page in enumerate(pdf.pages):
                print(f"\n处理第 {i + 1} 页...")
                try:
                    is_vertical_table = detect_table_orientation(page)
                    print(f"  表格方向: {'竖向' if is_vertical_table else '横向/无表格'}")

                    original_page = reader.pages[i]

                    if is_vertical_table:
                        print(f"  Page {i + 1}：旋转90度")
                        rotated_page = original_page.rotate(90)
                        writer.add_page(rotated_page)
                    else:
                        print(f"  Page {i + 1}：保持原样")
                        writer.add_page(original_page)
                    
                    successful_pages += 1
                        
                except Exception as page_error:
                    print(f"  处理第 {i + 1} 页时出错: {page_error}")
                    try:
                        writer.add_page(reader.pages[i])
                        successful_pages += 1
                    except Exception as backup_error:
                        print(f"  备用方案也失败: {backup_error}")
                        failed_pages += 1

            print(f"\n成功: {successful_pages} 页, 失败: {failed_pages} 页")

            if len(writer.pages) == 0:
                print("没有页面可写入，复制原文件")
                import shutil
                shutil.copy2(input_pdf_path, output_pdf_path)
                return
                
            with open(output_pdf_path, "wb") as output_pdf:
                writer.write(output_pdf)
            print(f"成功保存: {output_pdf_path}")
            
    except Exception as e:
        print(f"处理失败: {e}")
        import shutil
        try:
            shutil.copy2(input_pdf_path, output_pdf_path)
            print(f"已复制原文件到: {output_pdf_path}")
        except Exception as copy_error:
            print(f"复制文件失败: {copy_error}")

if __name__ == "__main__":
    input_pdf_paths = "dataset/rTMS/pdf"
    output_pdf_paths = "dataset/rTMS/pdf_processed"
    if not os.path.exists(output_pdf_paths):
        os.makedirs(output_pdf_paths)
    for input_pdf_path in os.listdir(input_pdf_paths):
        output_pdf_path = os.path.join(output_pdf_paths, input_pdf_path.split(".")[0] + "_processed.pdf")
        process_pdf(os.path.join(input_pdf_paths, input_pdf_path), output_pdf_path)