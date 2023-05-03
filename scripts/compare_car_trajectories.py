import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import rospkg

car_trj_sinusoids = np.load('car_trj_sinusoids.npy')
car_trj_trajopt = np.load('car_trj_trajopt.npy')

matplotlib.rcParams.update({'font.size': 45})

plt.figure(figsize=(20, 15))
plt.plot(car_trj_sinusoids[:, 0], car_trj_sinusoids[:, 1], linewidth=10, label='Sinusoids')
plt.plot(car_trj_trajopt[:, 0], car_trj_trajopt[:, 1], '--', linewidth=10, label='Trajectory Optimization')
plt.xlabel('x (m)', labelpad=100)
plt.ylabel('y (m)', labelpad=150, rotation=0)
ax = plt.gca()
ax.set_title('Comparison of Planned\nTrajectories for a Car', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.3, bottom=0.25, top=0.8)
plt.legend()
plt.savefig('compare_car_trajectories.png')
