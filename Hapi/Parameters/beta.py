"""
Created on Tue Mar 13 00:51:22 2018

@author: Mostafa
"""
#%links
# from IPython import get_ipython  # to reset the variable explorer each time

# get_ipython().magic("reset -f")
# import os
# import sys
from collections import OrderedDict

import matplotlib.pyplot as plt

#%%library
import numpy as np

# from matplotlib.transforms import blended_transform_factory

# os.chdir("")


# sys.path.append("")


linestyles = OrderedDict(
    [
        ("solid", (0, ())),
        ("loosely dotted", (0, (1, 10))),
        ("dotted", (0, (1, 5))),
        ("densely dotted", (0, (1, 1))),
        ("loosely dashed", (0, (5, 10))),
        ("dashed", (0, (5, 5))),
        ("densely dashed", (0, (5, 1))),
        ("loosely dashdotted", (0, (3, 10, 1, 10))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("densely dashdotted", (0, (3, 1, 1, 1))),
        ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
        ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ]
)

# functions


#%%
x = np.linspace(0, 1, 1000)
y1 = np.power(x, 0)
y2 = np.power(x, 0.2)
y3 = np.power(x, 0.5)
y4 = np.power(x, 0.8)
y5 = np.power(x, 1)

# y6=np.power(x,1.5)
y7 = np.power(x, 2)
# y8=np.power(x,3)
y9 = np.power(x, 4)
y10 = np.power(x, 6)
#%%plot
plt.figure(2, figsize=(8, 5))
plt.plot(x, y5, label="Beta=1", linewidth=5)
# plt.plot(x,y6,label="Beta=0.2",linewidth=5)
plt.plot(x, y7, label="Beta=2", linewidth=5)
# plt.plot(x,y8,label="Beta=0.5",linewidth=5)
plt.plot(x, y9, label="Beta=4", linewidth=5)
plt.plot(x, y10, label="Beta=6", linewidth=5)

plt.vlines(1, 0, 1, colors="#838B8B", linewidth=5)
plt.hlines(1, 0, 1, colors="#838B8B", linewidth=5)
plt.xlim([0, 1.05])
plt.ylim([0, 1.1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("SM/FC", fontsize=20)
plt.ylabel("(SM/FC)^beta", fontsize=20)
plt.legend(fontsize=20)
#%%
plt.figure(1, figsize=(8, 5))
plt.plot(x, y1, label="Beta=0", linewidth=5)
plt.plot(x, y2, label="Beta=0.2", linewidth=5)
plt.plot(x, y3, label="Beta=0.5", linewidth=5)
# plt.plot(x,y4,label="Beta=0.8",linewidth=5)
plt.plot(x, y5, label="Beta=1.0", linewidth=5)
plt.vlines(1, 0, 1, colors="#838B8B", linewidth=5)
plt.xlim([0, 1.05])
plt.ylim([0, 1.1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("SM/FC", fontsize=20)
plt.ylabel("(SM/FC)^beta", fontsize=20)
plt.legend(fontsize=30)

#%%
plt.figure(3, figsize=(10, 8))
# ax = plt.subplot(1, 1, 1)
plt.plot(x, y5, label="Beta=1", linewidth=5)
plt.annotate(r"$\beta=1 $", xy=(0.35, 0.48), fontsize=20, rotation=41)

plt.plot(x, y7, "--", label="Beta=2", linewidth=5)
plt.annotate(r"$\beta=2 $", xy=(0.53, 0.26), fontsize=20, rotation=38)

plt.plot(x, y10, ":", label="Beta=6", linewidth=5)
plt.annotate(r"$\beta=6 $", xy=(0.67, 0.12), fontsize=20, rotation=38)


plt.plot(x, y2, "-.", label="Beta=0.2", linewidth=5)
plt.annotate(r"$\beta=0.2 $", xy=(0.08, 0.75), fontsize=20, rotation=35)

plt.plot(
    x,
    y3,
    linestyle=linestyles.items()[4][1],
    label="Beta=0.5",
    linewidth=5,
    dashes=(10, 1),
)
plt.annotate(r"$\beta =0.5 $", xy=(0.22, 0.61), fontsize=20, rotation=37)

plt.vlines(1, -1, 1, colors="#838B8B", linewidth=5)
plt.hlines(1, -1, 1, colors="#838B8B", linewidth=5)
plt.xlim([0, 1.05])
plt.ylim([0, 1.1])

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$(\frac{SM}{FC})$", fontsize=25)
plt.ylabel(r"$(\frac{SM}{FC})^{\beta}$", fontsize=25)
# plt.legend(fontsize=20 ,frameon = False, loc = 8,ncol =2) #framealpha=1
