import matplotlib.pyplot as plt
import os.path
from scipy.io import loadmat
import numpy as np

# Result location
save_dir = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/tmp-sim/"
save_dir = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-26 3d vascular phantom (Francis' request)/3d/reconstruction-data/"

# Where to store images
# fig_dir = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-21 veins - improved frequency response (incl images)/reconstructions/"
# fig_dir = "//ad.utwente.nl/tnw/bmpi/data/mirre van der wal/data/2023-06-20 coregistered psf phantom (incl images)/reconstructions/"
fig_dir = "//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-26 3d vascular phantom (Francis' request)/3d/reconstructions/"
# fig_dir = "tmp"

NOISE = False

simulations = {
    "baseline": "baseline",
    "acoustic attenuation": "attenuation",
    "detector size": "size",
    "frequency response": "frequencyresponse",
    "acoustic attenuation + detector size": "attenuation_size",
    "acoustic attenuation + detector size + frequency response": "attenuation_size_frequencyresponse",
}

noise_str = "_noise" if NOISE else ""

for optical in [False, True]:
    optical_str = "optical_" if optical else ""
    for (title, filename) in simulations.items():
        if optical:
            title = "Optical + Acoustic:\n" + title
        else:
            title = "Acoustic:\n" + title

        full_filename = "sinogram_rekon_" + optical_str + filename + noise_str + ".mat"
        path = os.path.join(save_dir, full_filename)

        data = loadmat(path)["rekon"]

        plt.figure()
        plt.title(title)
        plt.imshow(data, extent=[0, 50, 50, 0], cmap="coolwarm")
        plt.xlabel("x-position [mm]")
        plt.ylabel("z-position [mm]")
        plt.colorbar(label="Pressure [a.u.]")

        plt.savefig(fig_dir + optical_str + filename + noise_str + ".png")

# experiment = loadmat("//ad.utwente.nl/TNW/BMPI/Data/Mirre van der Wal/Data/2023-06-09 psf data/psf_scan.mat")["P"]["reconEnv"][0][0][:, :, 4]
experimental_path = "//ad.utwente.nl/tnw/bmpi/Data/Mirre van der Wal/Data/2023-06-02 coregistered and normalized results/experimental.npy"
experiment = np.load(experimental_path)

experiment /= np.max(experiment)

plt.figure()
plt.title("Experimental")
plt.imshow(experiment, extent=[0, 50, 50, 0], cmap="coolwarm")
plt.xlabel("x-position [mm]")
plt.ylabel("z-position [mm]")
plt.colorbar(label="Pressure [a.u.]")

plt.savefig(fig_dir + "experimental.png")

# plt.show()
