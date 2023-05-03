import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

# I ran multiple trials for planning time
sinusoids_times = [0.000321, 0.000307, 0.000328]
trajopt_times = [4.604420, 4.567697, 4.552114]

names = ['Sinusoids', 'Trajectory Optimization']
values = np.array([np.mean(sinusoids_times), np.mean(trajopt_times)])

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.bar(names, values)
plt.xlabel('Planner', labelpad=100)
plt.ylabel('Planning\nTime (s)', labelpad=190, rotation=0)
ax.set_title('Planning Time Comparison\nfor a Hopper', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.3, bottom=0.4, top=0.8)
plt.savefig('compare_hopper_planning_time.png')
