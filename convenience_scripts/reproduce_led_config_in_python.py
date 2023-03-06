import numpy as np
import matplotlib.pyplot as plt

grid_size = 5.5e-2
TransFocus = 0
height = 5.5e-2
Nx = 744
dx = grid_size / Nx
full_angle = 120
n_photon = 1e6
laser_power = 1e-6
gamma = 0.15
illumination_type = 'side'
nm = 850
n_src = 4 * 36
height_probe = 0
n_angles = 1
height_phantom = 3.785e-2

x_vec = np.linspace(-Nx / 2, Nx / 2, Nx) * dx
y_vec = x_vec
Ny = Nx
Nz = int(height_phantom / dx)
z_vec = np.linspace(0, Nz, Nz) * dx

[_, _, Z] = np.meshgrid(x_vec, y_vec, z_vec)
mask = (Z <= height_phantom)

setting = dict()
setting["dx"] = dx
setting["Nx"] = round(Nx)
setting["Ny"] = round(Ny)
setting["Nz"] = round(Nz)
setting["Nxyz"] = [round(Nx), round(Ny), round(Nz)]
setting["N"] = np.prod([round(Nx), round(Ny), round(Nz)])
setting["x_vec"] = x_vec
setting["y_vec"] = y_vec
setting["z_vec"] = z_vec
setting["mask"] = mask


bar_length = 50e-3
dx_LEDs = 1.3511e-3
dz_LEDs = (6 / 3) * 1e-3
LED_casing = 10e-3
probe_casing = (12.16 / 2.0) * (10**-3)

x_row1 = np.linspace(-bar_length / 2 + dx_LEDs, bar_length / 2, 36)
x_row2 = np.linspace(-bar_length / 2, bar_length / 2 - dx_LEDs, 36)
x_bar = np.hstack([x_row1, x_row2, x_row1, x_row2])

Angle = 50
y_base = (TransFocus - 3e-3 - np.sin(np.deg2rad(Angle)) * 5e-3)
y_row1 = (y_base - (dz_LEDs * 1.5) * np.sin(np.deg2rad(Angle))) * np.ones((36, ))
y_row2 = (y_base - (dz_LEDs * 0.5) * np.sin(np.deg2rad(Angle))) * np.ones((36, ))
y_row3 = (y_base + (dz_LEDs * 0.5) * np.sin(np.deg2rad(Angle))) * np.ones((36, ))
y_row4 = (y_base + (dz_LEDs * 1.5) * np.sin(np.deg2rad(Angle))) * np.ones((36, ))
y_bar3 = np.hstack([y_row1, y_row2, y_row3, y_row4])

z_base = height_probe + 6e-3 + np.sin(np.deg2rad(Angle)) * 5e-3 / np.tan(np.deg2rad(Angle))
z_row1 = (z_base + (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row2 = (z_base + (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row3 = (z_base - (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row4 = (z_base - (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_bar3 = np.hstack([z_row1, z_row2, z_row3, z_row4])


vector = np.asarray([0, y_base, z_base])
v_bar1 = (vector / np.linalg.norm(vector) * np.ones((3, 144)).T).copy()

y_base = (TransFocus - 3e-3 - np.sin(np.deg2rad(Angle)) * 5e-3)
y_row1 = (y_base + (dz_LEDs * 1.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36, ))
y_row2 = (y_base + (dz_LEDs * 0.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36, ))
y_row3 = (y_base - (dz_LEDs * 0.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36, ))
y_row4 = (y_base - (dz_LEDs * 1.5) * np.sin(np.deg2rad(-Angle))) * np.ones((36, ))
y_bar4 = np.hstack([y_row1, y_row2, y_row3, y_row4])

z_base = height_probe - 5e-3 - np.sin(np.deg2rad(Angle)) * 5e-3 / np.tan(np.deg2rad(Angle))
z_row4 = (z_base + (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row3 = (z_base + (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row2 = (z_base - (dz_LEDs * 0.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_row1 = (z_base - (dz_LEDs * 1.5) * np.cos(np.deg2rad(Angle))) * np.ones((36, ))
z_bar4 = np.hstack([z_row1, z_row2, z_row3, z_row4])


vector = np.asarray([0, -y_base, -z_base])
v_bar2 = (vector / np.linalg.norm(vector) * np.ones((3, 144)).T).copy()

src = dict()
src["x_coordinates"] = np.asarray(np.hstack([x_bar, x_bar]))
src["y_coordinates"] = np.asarray(np.hstack([z_bar3, z_bar4]))
src["z_coordinates"] = np.abs(np.asarray(np.hstack([y_bar3, y_bar4])))
src["direction"] = np.asarray(np.vstack([v_bar1, v_bar2]))
src["numerical_apperature"] = full_angle
src["N"] = n_src
src["type"] = 'cone'

print(np.shape(src["x_coordinates"]))
print(np.shape(src["y_coordinates"]))
print(np.shape(src["z_coordinates"]))
print(np.shape(src["direction"]))

print(src["direction"])

plt.subplot(1, 1, 1, projection='3d')
plt.scatter(src["x_coordinates"], src["y_coordinates"], zs=src["z_coordinates"], marker="x", s=50, label='LED positions')

plt.plot(np.asarray([-50e-3 / 2, - 50e-3 / 2, 50e-3 / 2, 50e-3 / 2, - 50e-3 / 2]),
         np.asarray([height_probe - 5e-3, height_probe + 5e-3, height_probe + 5e-3, height_probe - 5e-3, height_probe - 5e-3]),
                    zs=TransFocus * np.ones((5, )),
                    c="r", linewidth=2, label='probe position')

plt.gca().quiver(src["x_coordinates"], src["y_coordinates"], src["z_coordinates"],
           src["direction"][:, 0], src["direction"][:, 1], src["direction"][:, 2],
                 label="Illumination direction", color="pink", alpha=0.5)
plt.xlim([setting["x_vec"][0], setting["x_vec"][-1]])
plt.ylim([setting["y_vec"][0], setting["y_vec"][-1]])
plt.gca().set_zlim([min(z_bar4), max(z_bar3)])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.gca().set_zlabel('z [m]')
plt.legend()
plt.show()