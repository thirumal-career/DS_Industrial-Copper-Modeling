import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d = {"hello":[1,2,3,4],"number":[11,12,13,14]}


df = pd.DataFrame(d)
column = "number"
sb.boxplot(data=d, x=column)
plt.show()


