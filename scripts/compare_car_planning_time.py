import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

# I ran multiple trials for planning time
sinusoids_times = [0.264259, 0.272711, 0.253241]
trajopt_times = [0.476372, 0.464175, 0.466406]

names = ['Sinusoids', 'Trajectory Optimization']
values = np.array([np.mean(sinusoids_times), np.mean(trajopt_times)])

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.bar(names, values)
plt.xlabel('Planner', labelpad=100)
plt.ylabel('Planning\nTime (s)', labelpad=190, rotation=0)
ax.set_title('Planning Time Comparison\nfor a Car', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.3, bottom=0.4, top=0.8)
plt.savefig('compare_car_planning_time.png')
