# Simple plotting check.
import braid
import braid_group
import numpy as np

g0 = braid_group.Generator(0) 
i1 = braid_group.InverseGenerator(1)
w2 = braid_group.Word(g0.Compose(i1).Compose(g0))
b2 = braid.Braid.Create(word=w2, num_strands=3)

ts = list(np.arange(0, 1.005, 0.005))
xs = {0: [], 1: [], 2: []}
ys = {0: [], 1: [], 2: []}

for t in ts:
  for s in range(3):
    p = b2.Strand(s).AtTime(t)
    xs[s].append(p[0])
    ys[s].append(p[1])

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs[0], ys[0], ts)
ax.plot(xs[1], ys[1], ts)
ax.plot(xs[2], ys[2], ts)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
plt.savefig('simple_plot.png')