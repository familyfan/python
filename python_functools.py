# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:56:54 2020

@author: 28369
"""

import functools


# =============================================================================
# def wrap_exception(func):
#     @functools.wraps(func)
#     def wrapper(self, *args, **kwargs):
#         try:
#             return func(self, *args, **kwargs)
#         except BaseException as ex:
#             raise MyException(ex.message)
# 
#     return wrapper
# 
# 
# class MyException(Exception):
#     def __init__(self, msg):
#         self.msg = msg
# 
#     def __str__(self):
#         return self.msg
# 
#     def __repr__(self):
#         return self.msg
# 
# 
# class Test:
#     def __init__(self):
#         pass
# 
#     @wrap_exception
#     def test(self):
#         raise Exception("hello")
# 
# 
# t = Test()
# t.test()
# 
# =============================================================================



# =============================================================================
# def wrap_exception(func):
#     @functools.wraps(func)
#     def wrapper(self, *args, **kwargs):
#         try:
#             return func(self, *args, **kwargs)
#         except:
#             pass
# 
#     return wrapper
# 
# 
# class Test:
#     def __init__(self):
#         pass
# 
#     @wrap_exception
#     def test(self):
#         raise Exception("hello")
# 
# 
# t = Test()
# t.test()
# 
# =============================================================================





# =============================================================================
# def wrap_logger(func):
#     @functools.wraps(func)
#     def wrapper(self, *args, **kwargs):
#         print ("%s(%s, %s)" % (func, args, kwargs))
#         print ("before execute")
#         result = func(self, *args, **kwargs)
#         print ("after execute")
#         return result
# 
#     return wrapper
# 
# 
# class Test:
#     def __init__(self):
#         pass
# 
#     @wrap_logger
#     def test(self, a, b, c):
#         print (a, b, c)
# 
# 
# t = Test()
# t.test(1, 2, 3)
# =============================================================================





import time

def wrap_performance(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        t_begin = time.time()
        result = func(self, *args, **kwargs)
        t_end = time.time()
        print("Time: %f " % (t_end - t_begin))
        return result

    return wrapper


class Test:
    def __init__(self):
        pass

    @wrap_performance
    def test(self):
        time.sleep(1)


t = Test()
t.test()



























