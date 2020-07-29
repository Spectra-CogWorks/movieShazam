from camera import take_picture
from PIL import Image
import numpy as np
from pathlib import Path


def import_image(file_path):
    """
    Loads a local image and converts it to an ndarray. 
    
    Parameters
    ----------
    file_path : String
        file path of the image
    
    Returns
    -------
    imgarray : numpy.ndarray
            A shape-(M,N,3) array that represents an M by N image of type .jpg, .png, or .jpeg.
    """
    # converts file_path to a Path object and opens it as an Image object
    path = Path(file_path)
    img = Image.open(path)
    # converts Image object to ndarray
    return np.asarray(img)


def import_folder(folder_path):
    """
    Loads all the .png, .jpg, and .jpeg files in a local folder and converts them to a list of ndarrays. 
    
    Parameters
    ----------
    folder_path : String
        folder path of the folder containing the images.
    
    Returns
    -------
    imglist : List[numpy.ndarray]
            A list of shape-(M,N,3) arrays that represents M by N images of type .jpg, .png, or .jpeg.
    """

    imglist = []
    # globs images from the given folder_path and adds them all to a list
    folder = Path(folder_path).glob("**/*.jpg")
    folder2 = Path(folder_path).glob("**/*.png")
    folder3 = Path(folder_path).glob("**/*.jpeg")
    folderlist = [folder, folder2, folder3]
    # iterates through the list, converts the images to ndarrays, and appends them to imglist
    for i in folderlist:
        for j in i:
            img = Image.open(j)
            imglist.append(np.asarray(img))
    # returns imglist
    return imglist
