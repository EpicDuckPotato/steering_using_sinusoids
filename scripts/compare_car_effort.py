import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

names = ['Sinusoids', 'Trajectory Optimization']
values = np.array([24.334929, 1.748224])

plt.figure(figsize=(20, 15))
ax = plt.gca()
plt.bar(names, values)
plt.xlabel('Planner', labelpad=100)
plt.ylabel('Effort', labelpad=190, rotation=0)
ax.set_title('Effort Comparison\nfor a Car', y=1.05)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.subplots_adjust(left=0.3, bottom=0.4, top=0.8)
plt.savefig('compare_car_effort.png')
