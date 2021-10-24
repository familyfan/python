import PyPDF2
from PyPDF2 import merger
import os
import glob


reader = PyPDF2.PdfFileReader(open("test.pdf", 'rb'))
# 获取pdf总页数
pages_num = reader.getNumPages()
# 判断是否有加密
is_encrypted = reader.isEncrypted
# 获取第四页
page = reader.getPage(4)
# 旋转180度
page.rotateClockwise(180)
# 获取第四页的内容
page_content = page.extractText()
# 获取PDF元信息，即包含创建时间，作者标题等。
doc_info = reader.getDocumentInfo()

"读取pdf文件，取指定页数，写入到硬盘上"
ouput = PyPDF2.PdfFileWriter()
output.addPage(reader.getPage(1))
output.addPage(reader.getPage(2))
output.addPage(reader.getPage(3))
# 获取写入页的总页数
write_pages_num = output.getNumPages()
# 加密
ouput.encrypt('family')
# 创建一个pdf文件
outputStream = open('pypdf2_output.pdf', 'wb')
# 往文件写入pdf数据
output.write(outputStream)
# 写入流
outputStream.close()



"合并多个指定pdf文件的示例"
## 创建一个合并的对象
merger = merger.PdfFileMerger()
input1 = open('pdf_test1.pdf', 'rb')
input2 = open('pdf_test2.pdf', 'rb')
input3 = open('pdf_test3.pdf', 'rb')

# 合并文件1中的0-3页
merger.append(fileobj=input1, pages=(0, 3))
# 合并文件2中的0-1页
merger.merge(position=2, fileobj=input2, pages=(0, 1))
# 合并文件的所有页
merger.append(fileobj=input3)
# 保存到硬盘上
output = open('merge_pdf_test1.pdf', 'wb')
# 写到硬盘上
merger.write(output)
# 关闭文件句柄
output.close()


"pdf增加水印示例"
reader = PyPDF2.PdfFileReader(open('linux.pdf', 'rb')) # 增加水印的原文件
 
watermark = PyPDF2.PdfFileReader(open('水印模板.pdf', 'rb')) # 水印的模板
 
writer = PyPDF2.PdfFileWriter() # 写入PDF的对象
 
for i in range(reader.getNumPages()):
  page = reader.getPage(i)
  page.mergePage(watermark.getPage(0)) # 将原文件与水印模板合并
  writer.addPage(page) # 增加到写入对象中
 
outputStream = open('watermark-test-linux.pdf', 'wb') # 打开一个写入硬盘的文件对象
writer.write(outputStream) # 将合并好的数据，写入硬盘中
outputStream.close() # 关闭文件句柄



"批量合并指定目录的pdf文件的示例"
def get_all_pdf_files(path):
  """获取指定目录的所有pdf文件名"""
  all_pdfs = glob.glob('{0}/*.pdf'.format(path))
  all_pdfs.sort(key=str.lower) # 排序
  return all_pdfs
 
def main():
  path = os.getcwd()
  all_pdfs = get_all_pdf_files(path)
  if not all_pdfs:
    raise SystemExit('没有可用的PDF类型文件')
 
  merger = PyPDF2.PdfFileMerger()
 
  first_obj = open(all_pdfs[0], 'rb') # 打开第一个PDF文件
  merger.append(first_obj) # 增加到合并的对象中
 
  file_objs = []
  for pdf in all_pdfs[1:]: # 读取所有的文件对象
    file_objs.append(open(pdf, 'rb'))
 
  for file_obj in file_objs:
    reader = PyPDF2.PdfFileReader(file_obj)
    merger.append(fileobj=file_obj, pages=(1, reader.getNumPages()))
 
  outputStream = open('merge-pdfs.pdf', 'wb')
  merger.write(outputStream)
  outputStream.close()
  for file_obj in file_objs: # 批量关闭文件句柄
    file_obj.close()












