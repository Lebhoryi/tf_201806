#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-25 12:45:19
# @Author  : Lebhoryi@gmail.com
# @Link    : http://example.org
# @Version : Birth weight data

import requests
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\'r\n')[5:]
# birth_header = [x for x in birth_data[0].split('') if len(x) >= 1]
birth_data = [[float(x) for x in y.split('') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
print(len(birth_data))
# print(len(birth_data[0]))



