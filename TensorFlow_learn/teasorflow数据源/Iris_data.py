#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-25 12:29:13
# @Author  : Lebhoryi@gmail.com
# @Link    : http://example.org
# @Version : Iris data,源自于<tensorflow机器实战>

import tensorflow as tf
from sklearn import datasets
iris = datasets.load_iris()
print(len(iris.data))
# 150,出现Deprecation Warning,弃用警告

print(len(iris.target))
# 150,出现Deprecation Warning,弃用警告

print(iris.target[0])
# 0,正常应该出现[5.1 3.5 1.4 0.2]

print(set((iris.target)))
# {0, 1, 2}




