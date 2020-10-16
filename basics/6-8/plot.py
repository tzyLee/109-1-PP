import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("time.csv", names=["bytes", "time"])

x = np.array(data["bytes"])
y = np.array(data["time"])
A = np.c_[x, np.ones(x.shape)]

m, c = np.linalg.lstsq(A, y, rcond=0)[0]
print(m, c)

data.plot(x="bytes", y="time", style="o")
plt.plot(x, m * x + c, "r")

plt.title("Average message passing time")
plt.xlabel("bytes")
plt.ylabel("time (sec)")
plt.legend(["raw data", f"fitted line: y = {m:4g}x + {c:4g}"])
plt.show()

print(f"lambda = {c:2g}, beta = {1/m:2g}")
