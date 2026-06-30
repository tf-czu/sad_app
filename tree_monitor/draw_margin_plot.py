import matplotlib.pyplot as plt


x_labels = ['0.1', '0.2', '0.4', '1']
y_values = [16.27, 8, 1.47, 0.13]

plt.rcParams['figure.figsize'] = [4, 3]
plt.bar(x_labels, y_values, color='steelblue', edgecolor='black', width=0.6)
plt.xlabel("Set limit (%)", fontsize=12)
plt.ylabel("Error percentage (%)", fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig("bar_margin.png", dpi=1200)
