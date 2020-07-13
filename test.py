# -*- coding: utf-8 -*-
import xlrd              #导入模块
data = xlrd.open_workbook('电影人.xlsx', encoding_override="utf-8")    #打开电影人.xlsx文件读取数据
table = data.sheets()[0]       #读取第一个（0）表单
#或者通过表单名称获取 table = data.sheet_by_name(u'Sheet1')
print(table.nrows)            #输出表格行数
print(table.ncols)            #输出表格列数
print(table.row_values(0))    #输出第一行
print(table.col_values(0))    #输出第一列
print(table.cell(0,2).value)  #输出元素（0,2）
