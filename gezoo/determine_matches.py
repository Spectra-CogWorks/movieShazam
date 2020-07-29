import numpy as np
from database import Database  # pylint: disable=import-error


def determine_matches(fingerprints, threshold=2):
    """
    Determines the best match out of the names in the database for each input fingerprint.

    Parameter:
    -----------
    fingerprints: np.ndarray
        A shape-(N, 512) array of fingerprints (or descriptor vectors) taken from N images.

    threshold : float
        The threshold value for standard deviations

    Returns:
    --------
    matches: List[str]
        A list of the names whose fingerprints have the smallest cosine distance
    to each input/new fingerprint, i.e. are the best matches.
    """
    db = Database()
    matches = []

    # Loops over all N fingerprints in the input array
    for i in range(fingerprints.shape[0]):
        name_dists = []
        mean_dists = []

        for name in db.database.keys():
            dists = []

            # Each name contains multiple fingerprints
            for f in db.get_fingerprints(name):
                # Takes the cosine distances between input and database fingerprints for this name
                diff = cosine_distance(fingerprints[i].reshape((512)), f.reshape((512)))

                dists.append(diff)

            # computes mean distance and appends it to "name_dists"
            name_dists.append(name)

            # appends mean distance for each name to "mean_dists"
            mean_dists.append(np.mean(dists))

        # appends the name with the lowest mean distance to list "matches" if it falls within 2 stds
        print(mean_dists)
        print(np.std(mean_dists))
        if np.min(mean_dists) <= threshold:
            matches.append(name_dists[np.argmin(mean_dists)])
        else:
            matches.append("Unknown")

    db.save()

    return matches


def cosine_distance(d1, d2):
    """
    Finds the cosine distance between two arrays.
    
    Parameters:
    -----------
    d1: np.ndarray
    One of the two arrays you are finding the distance between.
    
    d2: np.ndarray
        The second of the two arrays you are finding the distance between.
            
    Returns:
    --------
    float
        The cosine distance between the two arrays.
    """
    return 1 - ((np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
