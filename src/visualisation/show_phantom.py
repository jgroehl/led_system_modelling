import matplotlib.pyplot as plt
import numpy as np

# Name of the phantom to inspect
PHANTOM = "veins"

label_mask = np.load("../../resources/phantoms/" + PHANTOM + ".npz")["gt"]
plt.figure()
plt.imshow(label_mask, extent=[0, 50, 50, 0])
plt.title("Ground truth phantom")
plt.xlabel("x-position [mm]")
plt.ylabel("z-position [mm]")
plt.colorbar(label="Pressure [a.u.]")

plt.show()
