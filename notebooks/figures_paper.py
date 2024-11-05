import matplotlib.pyplot as plt

# Data
methods = ['VXM', 'VTN', 'TSM', 'Dual-Encoder']
memory_usage = [309.17, 620.20, 450.76, 230.12 ]

# Create a dot plot
plt.figure(figsize=(6, 4))
plt.scatter(methods, memory_usage, color='blue', s=100, marker='o')
plt.title("Memory Usage of Deep Learning Models")
plt.xlabel("Models")
plt.ylabel("Memory (MB)")
plt.ylim(0, max(memory_usage) + 100)  # Set y-axis limit to give some space above the highest dot

# Adding memory usage as labels above each dot
for i, (method, mem) in enumerate(zip(methods, memory_usage)):
    plt.text(i, mem + 10, f"{mem} MB", ha="center", fontsize=9)

plt.tight_layout()

# Save the figure as a .png file
plt.savefig("memory_usage_dot_plot.png")
plt.close()


# Data
methods = ['VXM', 'VTN', 'TSM', 'CLMorph']
memory_usage = [309.17, 620.20, 450.76, 230.12 ]

# Create a dot plot
plt.figure(figsize=(6, 4))
plt.scatter(methods, memory_usage, color='blue', s=100, marker='o')
#plt.title("Memory Usage of Deep Learning Registration Models")
plt.xlabel("Models")
plt.ylabel("Memory (MB)")
plt.ylim(0, max(memory_usage) + 100)  # Set y-axis limit to give some space above the highest dot

# Adding memory usage as labels above each dot
for i, (method, mem) in enumerate(zip(methods, memory_usage)):
    plt.text(i, mem + 10, f"{mem} MB", ha="center", fontsize=9)

plt.tight_layout()

# Save the figure as a .png file
plt.savefig("memory_usage_dot_plot.png")
plt.close()



# Data
methods = ['ANTs', 'Elastix', 'VXM', 'VTN', 'TSM', 'CLMorph']
time_usage = [99.83, 128.77, 0.20, 0.14, 0.31, 0.10 ]

# Create a dot plot
plt.figure(figsize=(6, 4))
plt.scatter(methods, time_usage, color='blue', s=100, marker='o')
#plt.title("Time of Inference ")
plt.xlabel("Models")
plt.ylabel("Time (s)")
plt.ylim(0, max(time_usage) + 100)  # Set y-axis limit to give some space above the highest dot

# Adding memory usage as labels above each dot
for i, (method, mem) in enumerate(zip(methods, time_usage)):
    plt.text(i, mem + 10, f"{mem} s", ha="center", fontsize=9)

plt.tight_layout()

# Save the figure as a .png file
plt.savefig("time_usage_dot_plot.png")
plt.close()