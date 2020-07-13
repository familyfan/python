# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:27:57 2020

@author: 28369
"""

import argparse
 
parser = argparse.ArgumentParser()
 
parser.add_argument("--square", help="display a square of a given number", type=int)
parser.add_argument("--cubic", help="display a cubic of a given number", type=int)
 
args = parser.parse_args()
 
if args.square:
    print(args.square**2)
 
if args.cubic:
    print(args.cubic**3)
    
#将上面的代码保存为文件argparse_usage.py, 在终端运行，结果如下：
#$ python argparse_usage.py --h
#usage: argparse_usage.py [-h] [--square SQUARE] [--cubic CUBIC]
# 
#optional arguments:
#  -h, --help       show this help message and exit
#  --square SQUARE  display a square of a given number
#  --cubic CUBIC    display a cubic of a given number
# 
#$ python argparse_usage.py --square 8
#64
# 
#$ python argparse_usage.py --cubic 8
#512
# 
#$ python argparse_usage.py 8
#usage: argparse_usage.py [-h] [--square SQUARE] [--cubic CUBIC]
#argparse_usage.py: error: unrecognized arguments: 8
# 
#$ python argparse_usage.py  # 没有输出
    
    
    
    
    
    
    
    
    
    
    
    
    