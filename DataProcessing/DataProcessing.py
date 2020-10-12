import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


label_file_path = './Data/Pedestrian_Recognition/train/label.txt'
with open(label_file_path, 'r') as label_file:
    train_labels = label_file.readlines()

index_dict = {}
for train_label in train_labels:
    index = (int)(train_label.strip().split(":")[-1])
    index_dict.setdefault(index, [0])
    index_dict[index][0] += 1

label_frame = pd.DataFrame.from_dict(index_dict, orient='index')
plt.figure()
sns.histplot(data=label_frame, stat="count", discrete=True)
plt.xlim(10, 810)
plt.ylim(0, 10)
pic_file = 'Pictures/TenToInf.png'
plt.savefig(pic_file)
plt.show()
print(label_frame.describe(include="all"))

