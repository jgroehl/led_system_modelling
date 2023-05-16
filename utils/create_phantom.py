from pandas import read_csv
import numpy as np

CSV_LOCATION = "../resources/disc.csv"
PHANTOM_NAME = "disc"


def create_phantom(name: str, csv_path: str) -> None:
    """
    Read a csv file with a pressure distribution, and export this to a .npz archive in the 'phantoms' folder.

    :param str name: The name of the phantom, will be used to name the created phantom .npz file
    :param str csv_path: The path to the csv file with the pressure distribution of the phantom
    """

    df = read_csv(csv_path)
    path = "../phantoms/" + name + ".npz"
    np.savez(path, gt=df)


if __name__ == "__main__":
    create_phantom(PHANTOM_NAME, CSV_LOCATION)
