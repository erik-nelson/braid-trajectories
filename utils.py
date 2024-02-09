import braid
import braid_group
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple

# Generates a set of agent trajectories for a given input braid. The output is an array of size:
#    num_agents x num_timestamps x 2
# where output[a][t][i] accesses the i'th coordinate at the t'th timestamp of the a'th agent.
#
# Internally we use `num_segments` to rescale time. In general a braid is built by repeatedly
# concatenating a set of functions defined on the unit interval [0, 1] in time. After one
# such composition, the interval [0, 0.5] would sample the first function and the interval
# [0.5, 1] would sample the second. After two such compositions the first two intervals get
# "squished", i.e. [0, 0.25], [0.25, 0.5], and [0.5, 1]. Each additional function composition
# squishes the previous intervals, reducing their time by half. In order to evenly sample
# each function, we invert this scaling in this function, making sure to sample the same number 
# of timestamps in each of [0.5, 1], [0.25, 0.5], [0.125, 0.25], [0.0625, 0.125], ...
def BraidToTrajectory(braid: braid.Braid, 
                      num_timestamps: int = 10, 
                      num_segments: int = 1) -> List[List[Tuple[float, float]]]:
  # Generate time intervals for `num_segments` segments.
  intervals = []
  start, end = 0, 1
  for _ in range(num_segments - 1):
    midpoint = (start + end) / 2
    intervals.append([midpoint, end])
    end = midpoint
  intervals.append([0, end])
  intervals.reverse()
  timestamps_per_interval = int(np.ceil(num_timestamps / len(intervals)))

  # Generate one trajectory per braid strand.
  trajectories = []
  num_agents = len(braid.strands)
  for i in range(num_agents):
    trajectory = []

    # Sample an even number of timestamps from each interval.
    for interval in intervals:
      for t in np.linspace(interval[0], interval[1], timestamps_per_interval, endpoint=False):
        position = braid.Strand(i).AtTime(t)
        trajectory.append((position[0], position[1]))

    # Always sample the endpoint of the braid as well.
    position = braid.Strand(i).AtTime(1)
    trajectory.append((position[0], position[1]))

    trajectories.append(trajectory)

  return trajectories

# Plot a set of input trajectories to an output file. This produces a 2x2 grid of subplots showing
# various cross sections of the (x, y, t) input trajectories.
def PlotTrajectories3D(trajectories: List[List[Tuple[float, float]]], 
                       save_file: str = 'braid_3d.png'):
  num_timestamps = len(trajectories[0])
  ts = np.linspace(0, 1, num_timestamps)
  num_agents = len(trajectories)
  xs = {agent_idx: [] for agent_idx in range(num_agents)}
  ys = {agent_idx: [] for agent_idx in range(num_agents)}

  for agent in range(num_agents):
    for position in trajectories[agent]:
      xs[agent].append(position[0])
      ys[agent].append(position[1])

  fig = plt.figure()
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
  colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

  # Set up 2D x-t cross section plot (top left corner).
  ax_xt = axes[0, 0]
  for i in range(num_agents):
    ax_xt.plot(xs[i], ts, color=colors[i])
  ax_xt.set_xlabel('X')
  ax_xt.set_ylabel('Time')
  ax_xt.grid(True)
  plt.axis('equal')

  # Set up 2D y-t cross section plot (top right corner).
  ax_yt = axes[0, 1]
  for i in range(num_agents):
    ax_yt.plot(ys[i], ts, color=colors[i])
  ax_yt.set_xlabel('Y')
  ax_yt.set_ylabel('Time')
  ax_yt.grid(True)
  plt.axis('equal')

  # Set up 2D x-y cross section plot (bottom right corner).
  ax_xy = axes[1, 0]
  for i in range(num_agents):
    ax_xy.plot(xs[i], ys[i], color=colors[i])
  ax_xy.set_xlabel('X')
  ax_xy.set_ylabel('Y')
  ax_xy.grid(True)
  plt.axis('equal')

  # Set up 3D plot (bottom right corner).
  axes[1, 1].set_visible(False)
  ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
  for i in range(num_agents):
    ax_3d.plot(xs[i], ys[i], ts, color=colors[i])
  ax_3d.set_xlabel('X')
  ax_3d.set_ylabel('Y')
  ax_3d.set_zlabel('Time')
  ax_3d.set_zlim([0, 1])
  ax_3d.set_box_aspect([1, 1, 3])

  plt.tight_layout()
  plt.savefig(save_file)
  plt.close()

def AnimateTrajectories(trajectories: List[List[Tuple[float, float]]], 
                        save_file: str = 'trajectories.gif'):
  fig, ax = plt.subplots()
  num_agents = len(trajectories)
  colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
  ax.grid(True)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')

  # Set axis limits dynamically.
  positions = [position for trajectory in trajectories for position in trajectory]
  min_x = min(position[0] for position in positions)
  max_x = max(position[0] for position in positions)
  min_y = min(position[1] for position in positions)
  max_y = max(position[1] for position in positions)
  ax.set_xlim(min_x - 0.1, max_x + 0.1)
  ax.set_ylim(min_y - 0.1, max_y + 0.1)

  # Initialize empty plot objects for trajectories.
  trajectory_plots = [ax.plot([], [], color=colors[i])[0] for i in range(num_agents)]

  # Function to update the plot for each frame.
  def update(frame):
    for i, trajectory in enumerate(trajectories):
        x, y = zip(*trajectory[:frame+1])  # Unzip x, y positions
        trajectory_plots[i].set_data(x, y)
    return trajectory_plots

  # Create the animation object.
  num_frames = len(trajectories[0])
  ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

  # Save the animation as a GIF.
  ani.save(save_file, writer='imagemagick')
  plt.close()