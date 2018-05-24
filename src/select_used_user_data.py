import numpy as np
import pandas as pd


path = '../data'
cnt = 1
fw = open(path + 'select_500.data', 'w')
with open(path + '/userFeature.data') as f:
    for e in f:
        print(e)
        fw.write(e)

        if cnt > 500:
            break
        cnt += 1
fw.close()
