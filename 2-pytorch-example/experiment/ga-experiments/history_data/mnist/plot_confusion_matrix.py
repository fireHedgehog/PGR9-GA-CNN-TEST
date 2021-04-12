import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

array = np.array([[965, 2, 0, 0, 2, 0, 6, 0, 22, 0],
                  [0, 1064, 8, 0, 8, 0, 9, 0, 15, 0],
                  [239, 14, 574, 0, 18, 0, 34, 0, 39, 0],
                  [496, 41, 201, 21, 18, 0, 4, 0, 242, 2],
                  [290, 17, 0, 0, 626, 0, 22, 0, 42, 0],
                  [499, 24, 12, 0, 144, 0, 13, 0, 213, 1],
                  [164, 34, 12, 0, 27, 0, 680, 0, 91, 0],
                  [332, 86, 126, 0, 91, 0, 0, 11, 208, 175],
                  [161, 43, 29, 0, 70, 0, 12, 0, 663, 0],
                  [330, 29, 2, 0, 428, 0, 2, 0, 239, 8]])

df_cm = pd.DataFrame(array, index=[i for i in "0123456789"],
                     columns=[i for i in "0123456789"])

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.show()
