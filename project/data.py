import pandas as pd
import random

data = pd.read_csv('tmalign.out', names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')

# 这里没有去重，如果实际训练的话建议去除重复的name
names = data['name1'].tolist() + data['name2'].tolist()