# -*- coding: utf-8 -*-
import h5py
import numpy as np
"创建dataset"
#f=h5py.File("h5py.hdf5","w")

#分别创建dset1,dset2,dset3这三个数据集
#a=np.arange(20)
#d1=f.create_dataset("dset1",data=a)
#
#d2=f.create_dataset("dset2",(3,4),'i')
#d2[...]=np.arange(12).reshape((3,4))
#
#f["dset3"]=np.arange(15)
#
#for key in f.keys():
#    print(f[key].name)
#    print(f[key].value)



#/dset1
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
#/dset2
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
#/dset3
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    
    
"创建group组"
#f=h5py.File("test.hdf5","w")
#
##创建组bar1,组bar2，数据集dset
#g1=f.create_group("bar1")
#g2=f.create_group("bar2")
#d=f.create_dataset("dset",data=np.arange(10))
#
##在bar1组里面创建一个组car1和一个数据集dset1。
#c1=g1.create_group("car1")
#d1=g1.create_dataset("dset1",data=np.arange(10))
#
##在bar2组里面创建一个组car2和一个数据集dset2
#c2=g2.create_group("car2")
#d2=g2.create_dataset("dset2",data=np.arange(10))
#
##根目录下的组和数据集
#print(".............")
#for key in f.keys():
#    print(f[key].name)
#
##.............
##/bar1
##/bar2
##/dset
#
##bar1这个组下面的组和数据集
#print(".............")
#for key in g1.keys():
#    print(g1[key].name)
#
##.............
##/bar1/car1
##/bar1/dset1
#
##bar2这个组下面的组和数据集
#print(".............")
#for key in g2.keys():
#    print(g2[key].name)
#
##.............
##/bar2/car2
##/bar2/dset2
#
##顺便看下car1组和car2组下面都有什么，估计你都猜到了为空。
#print(".............")
#print(c1.keys())
#print(c2.keys())
#
##.............
##KeysView(<HDF5 group "/bar1/car1" (0 members)>)
##KeysView(<HDF5 group "/bar2/car2" (0 members)>)    
    
    
    