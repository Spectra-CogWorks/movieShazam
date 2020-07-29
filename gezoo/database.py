import pickle
import numpy as np
from pathlib import Path
from copy import deepcopy
import model_wrapper as mw


class Database:
    def __init__(self, importVal=True, path="./database.pkl"):
        """Initialize a database, either fresh or imported
        
        Parameters
        ----------
        import : boolean = True
            Choose whether there would should be a fresh dictionary or imported from a file
            
        path : str
            A path to the database path
        """
        if importVal:
            self.database = pickle.load(open(path, "rb"))
        else:
            self.database = {}

    def save(self, path="./database.pkl"):
        """Saves a database returned from default

        Parameters
        -------
        database: Dict{str : Profile]
            Single database instance containing profiles.
            
        path : str
            The path of the database
        """
        datab = deepcopy(self.database)

        with open(path, mode="wb") as opened_file:
            pickle.dump(datab, opened_file)

    def load(self, path="./database.pkl"):
        """Loads a database

        Parameters
        -------
        path : str
            The path of the database
        
        Returns
        -------
        database : Dict{str : Profile}
        """
        return pickle.load(open(path, "rb"))

    def add(self, name, fingerprint):
        """Adds a fingerprint to a specific database entry
        
        Parameters
        ----------
        name : str
            The name key of the entry
            
        fingerprint : np.ndarray - shape(512,)
            The fingerprint being added to the database
        """
        if name not in self.database:
            self.database[name] = [fingerprint]
        else:
            self.database[name].append(fingerprint)

    def add_multi(self, name, fingerprints):
        """Adds a fingerprint to a specific database entry
        
        Parameters
        ----------
        name : str
            The name key of the entry
            
        fingerprints : List[np.ndarray - shape(512,)]
            The fingerprints being added to the database
        """
        if name not in self.database:
            self.database[name] = fingerprints
        else:
            self.database[name].extend(fingerprints)

    def get_fingerprints(self, name):
        """The fingerprints are returned for a specific name
        
        Parameters
        ----------
        name : str
            The key of the database
            
        Returns
        -------
        self.database[name] : List[np.ndarray - shape(512,)]
        """
        return self.database[name]

    @classmethod
    def compute_fingerprint_from_image(cls, img):
        """Compute a fingerprint from an image
        
        Parameters
        ----------
        img : np.ndarray
            An image
            
        Returns
        -------
        fingerprints : np.ndarray - shape(512,)
        """

        boxes = mw.feed_mtcnn(img)

        #checks if there are boxes to compute. If not, then returns none
        if len(boxes)>=1:
            return mw.compute_fingerprints(img, boxes)
        else:
            return None

