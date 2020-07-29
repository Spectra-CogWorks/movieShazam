import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def display_image(image, boxes=None, names=None, fontsize=20):
    """
    Displays an image of people. These people have boxes surrounding them
    and their names depicted next to them.

    Parameters
    ----------
    image : np.ndarray, shape=(H, W, C)
        The np.array that holds the image data. Where H is the height of the pic, W is the width, and C is the color channels.
        
    boxes : np.ndarray, shape(N, 4)
        The boxes around the faces
        
    name : List[str]
        An optional argument to add corresponding names
        
    fontsize : int
        An optional argument to adjust fontsize
        
    Returns
    -------
    None
    """
    # Divider
    if boxes is not None:
        fig, ax = plt.subplots()  # pylint: disable=unused-variable
        ax.imshow(image)

        for i in range(len(boxes)):
            ax.add_patch(
                Rectangle(
                    xy=(boxes[i][0], boxes[i][1]),
                    height=boxes[i][3] - boxes[i][1],
                    width=boxes[i][2] - boxes[i][0],
                    fill=None,
                    lw=2,
                    color="red",
                )
            )

            if names is not None:
                plt.text(
                    boxes[i][0], boxes[i][1], names[i], fontsize=fontsize, color="green"
                )

        plt.show(block=True)
    else:
        fig, ax = plt.subplots()  # pylint: disable=unused-variable
        ax.imshow(image)

        plt.show(block=True)
