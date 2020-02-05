import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['axes.unicode_minus'] = False

n_bins = 5

fig, ax = plt.subplots(figsize=(8, 5))

x_multi = [np.full(j, i) for i,j in zip([0.5, 0.6, 0.7, 0.8, 0.9], [1, 2, 3, 4, 5])]
from IPython import embed;embed()
ax.hist(x_multi, n_bins, histtype='bar', label=["NMS", "BM-NMS"])

ax.set_title('NMS v.s BM-NMS')

ax.legend()

plt.savefig('NMS-compare')
