import pandas as pd
import pathlib
import numpy as np


def load_image(image_file_name: str, dataset_images_folder: pathlib.Path, file_format: str = ".npy") -> np.ndarray:
    """
    Function that loads an image saved in a file as a numpy array.

    Parameters
    ----------
    image_file_name: str
        The name of the file containing the image data.
    dataset_images_folder: pathlib.Path
        The path of the folder containing the images.
    file_format: str
        The file format of the image files. Defaults to ".npy".

    Returns
    -------
    np.ndarray
        The numpy array containing the image data obtained from the file.
    """

    return np.load(dataset_images_folder.joinpath(image_file_name + file_format))


def load_dataset(dataset_folder: str, dataset_file: str = "inference-data.csv",
                 dataset_images_folder: str = "image-data") -> pd.DataFrame:
    """
    Method that loads a dataset dataframe, also adding image data.

    Parameters
    ----------
    dataset_folder: str
        The path of the folder containing the dataset data.
    dataset_file: str
        The name of the file containing the dataset data.
    dataset_images_folder: str
        The name of the folder containing the image data.

    Returns
    -------
    pd.Dataframe
        A dataframe of the dataset loaded from file and with added image data.
    """

    # define dataset paths
    dataset_folder = pathlib.Path(dataset_folder)
    dataset_df_path = dataset_folder.joinpath(dataset_file)
    dataset_images_folder = dataset_folder.joinpath(dataset_images_folder)

    # load the csv file containing information about the bounding boxes
    dataset_df = pd.read_csv(dataset_df_path)

    # pack bounding boxes coordinates
    dataset_df["image-data"] = dataset_df["nodule-image-name"].apply(load_image, args=(dataset_images_folder,))

    # remove not useful columns
    dataset_df = dataset_df.drop(
        columns=["scan-name", "nodule-image-size-x", "nodule-image-size-y", "nodule-image-name"])

    return dataset_df
