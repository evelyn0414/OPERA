# -*- coding: utf-8 -*-

''
# load labels and save training and test csv

''


import os

metacsv = open('metadata.csv', 'w')

metacsv.write('filename,split,label' + '\n')

path = 'HF_Lung_V1-master/train'

folders = os.listdir(path)

for file in folders:
    wheeze = False 
    stridor = False 
    rhonchi = False
    
    label = 'None'
    if '.txt' in file:
        print(file)
        f = open(path + '/' + file, 'r')
        content = f.read()
        
        if 'Wheeze' in content:
            wheeze = True
            label = 'Wheeze'
        
        if 'Stridor' in content:
            stridor = True
            label = 'Stridor'
        
        if 'Rhonchi' in content:
            rhonchi = True
            label = 'Rhonchi'
        
        if int(wheeze) + int(stridor) + int(rhonchi) > 1:
            label = 'Both'
            
        metacsv.write( file.split('.')[0][:-6] + ',' + 'train,' + label + '\n')
            
path = 'HF_Lung_V1-master/test'

folders = os.listdir(path)

for file in folders:
    wheeze = False 
    stridor = False 
    rhonchi = False
    
    label = 'None'
    if '.txt' in file:
        
        print(file)
        f = open(path + '/' + file, 'r')
        content = f.read()
        
        if 'Wheeze' in content:
            wheeze = True
            label = 'Wheeze'
        
        if 'Stridor' in content:
            stridor = True
            label = 'Stridor'
        
        if 'Rhonchi' in content:
            rhonchi = True
            label = 'Rhonchi'
        
        if int(wheeze) + int(stridor) + int(rhonchi) > 1:
            label = 'Both'
            
        metacsv.write( file.split('.')[0][:-6] + ',' + 'test,' + label + '\n')
                        
            
            

        


        


