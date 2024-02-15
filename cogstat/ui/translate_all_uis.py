# -*- coding: utf-8 -*-
import os
#print os.listdir('.')
for file in os.listdir('.'):
    if file[-2:]=='ui':
        command = 'pyuic6 -x %s > %s.py'%(file, file[:-3])
        print (command)
        os.system(command)

