# -*- coding: utf-8 -*-
import glob

# 查找符合特定规则的文件路径名
#三个匹配符: " ", "?", "[]"
#" " : 匹配零个或多个字符;
#"?" : 匹配单个字符;
#"[]" : 匹配指定范围内的字符, 如[0-9]匹配数字.



#glob.glob
#参数为相对路径“./*”或者“*”获得所有文件路径。如果文件名是已知的，那么参数为“./filename”或者“filename”

## 获取指定目录下的所有图片
## r表示让字符串不转义
#print(glob.glob(r"./*.jpg"), "\n")
##['.\\200_1_0.jpg', '.\\dog.jpg', '.\\fig_1.jpg', '.\\fig_q30.jpg', 
## '.\\fig_q60.jpg', '.\\fig_q90.jpg', '.\\lean.jpg']



















