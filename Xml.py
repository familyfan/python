# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

tree = ET.parse("test.xml")
#获取根节点
root = tree.getroot()
#tag = root.tag
#print(tag)  #students
#获取根节点属性
#print(root.attrib) #{}
#获取子节点的名称和属性
for child in root:
    print(child.tag, child.attrib)
#student {'no': '2009081097'}
#student {'no': '2009081098'}
#student {'no': '2009081099'}
    
##获取属性对应的值
#for student in root.findall("student"):
#    no = student.get("no")
#    name = student.find("name").text
#    print(no,  name)
##2009081097 Hongten
##2009081098 DuDu
##2009081099 Sum
    
#修改xml文件内容
#for age in root.iter("age"):
#    new_age = int(age.text) + 1
#    age.text = str(new_age)
#    age.set("updated", "yes")
#tree.write("test_update.xml")
    

"""
<students>
    <student no="2009081097">
        <name>Hongten</name>
        <gender>M</gender>
        <age>20</age>
        <score subject="math">97</score>
        <score subject="chinese">90</score>
    </student>
    <student no="2009081098">
        <name>DuDu</name>
        <gender>W</gender>
        <age>21</age>
        <score subject="math">87</score>
        <score subject="chinese">96</score>
    </student>
    <student no="2009081099">
        <name>Sum</name>
        <gender>M</gender>
        <age>19</age>
        <score subject="math">64</score>
        <score subject="chinese">98</score>
    </student>
</students>
"""
























