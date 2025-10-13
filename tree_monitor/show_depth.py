import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit(1)

filename = sys.argv[1]
data = np.load(filename)

print("Arays available in the file:", list(data.keys()))
arr = data["depth"]

arr[arr>3000] = 0
plt.imshow(arr, cmap='gray')
plt.title(f"depth from {filename}")
plt.colorbar()
plt.show()
