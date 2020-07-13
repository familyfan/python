# -*- coding: utf-8 -*-
import re
"re.match"
#re.match(pattern, string, flags=0) 
#patter:匹配的正则表达式
#string:要匹配的字符串
#flags:标志位, 用于控制正则表达式的匹配方式，
#如: 是否区分大小写，多行匹配等
#匹配成功返回一个匹配的对象，否则返回None.
#print(re.match("www", "www.runoob.com").span())  # 在起始位置匹配
#print(re.match("com", "www.runoob.com"))         # 不在起始位置匹配
#(0, 3)
#None

"re.search"
#re.search(pattern, string, flags=0)
#print(re.search("www", "www.runoob.com").span())
#print(re.search("com", "www.runoob.com").span())
#(0, 3)
#(11, 14)

"re.sub"
#re.sub(pattern, repl, string, count=0, flags=0)
#pattern : 正则中的模式字符串。
#repl : 替换的字符串，也可为一个函数。
#string : 要被查找替换的原始字符串。
#count : 模式匹配后替换的最大次数，
#默认 0 表示替换所有的匹配。
phone = "2004-959-559 # 这是一个国外电话号码"
#删除字符串中python注释
num = re.sub(r"#.*", "", phone)
print("电话号码是:", num)
#删除非数字(-)的字符串
num = re.sub(r"\D", "", phone )
print("电话号码是:", num)
"""
电话号码是: 2004-959-559 
电话号码是: 2004959559
"""



