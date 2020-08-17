import csv
import json
import ast
from PIL import Image
import numpy as np
from pathlib import Path
import determine_matches as dm
from database import Database as db
from determine_matches import determine_matches
from camera_input import import_image
from image_display import display_image
import model_wrapper as mw
import pickle
def actorsFromImage(file_path):
    img = import_image(file_path)
    boxes = mw.feed_mtcnn(img)
    fp=[]
    if len(boxes)>=1:
        fp = mw.compute_fingerprints(img, boxes)
    names = determine_matches(fp,threshold=1)
    display_image(img,boxes=boxes,names=names)
    print(names)
    return names
def determine_movies(listOfActors, threshold = 0.75):
    matches = []
    length = 1
    for i in maindict.keys():
        if set(listOfActors) & set(maindict[i][0])==set(listOfActors):
            matches.append(i)
    return matches
def movieShazam(filepath):
    actors = actorsFromImage(filepath)
    matches = determine_movies(actors)
    print (matches)