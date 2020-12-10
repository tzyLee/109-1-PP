import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("comb.csv", names=["p", "time"])

grouped = (
    df.groupby("p")["time"]
    .apply(lambda df: df.reset_index(drop=True))
    .unstack()
    .transpose()
)

arr = grouped.to_numpy()

for i in range(8):
    col = arr[:, i]
    print(i + 1, np.sort(col)[:25].mean())

for i in range(8):
    # plt.plot(np.sort(arr[:, i]))
    plt.plot(arr[:, i])
plt.legend(["p={}".format(i + 1) for i in range(8)])
plt.ylabel("time")
plt.xlabel("nth run")
plt.show()
