import click as ck
from pathlib import Path
import matplotlib.pyplot as plt

from .database import Database
from . import camera_input as inp
from . import model_wrapper as mw
from . import determine_matches as dm
from . import image_display as imd
from . import whispers_algorithm as whisp
from . import node as nd
import camera as cam


@ck.group()
def cli():
    pass


@cli.command()
def peek():
    db = Database()
    ck.echo(db.database)
    
    
@cli.command()
def clear():
    db = Database(importVal=False)
    db.save()
    ck.echo("Database Cleared!")


@cli.command()
@ck.argument("name")
@ck.argument("filepath")
@ck.option("-d", "--dbpath", help="Path to database")
@ck.option(
    "-p",
    "--probabilitythreshold",
    help="The threshold for when faces are identified",
    type=ck.FLOAT,
    default=0.8,
)
def add_file_to_database(name, filepath, probabilitythreshold, dbpath):
    db = Database()

    img = inp.import_image(filepath)

    fingerprints = mw.compute_fingerprints(
        img, mw.feed_mtcnn(img, threshold=probabilitythreshold)
    )

    db.add_multi(name, list(fingerprints))
    db.save()


@cli.command()
@ck.option("-d", "--dbpath", help="Path to database")
@ck.option(
    "-p",
    "--probabilitythreshold",
    help="The threshold for when faces are identified",
    type=ck.FLOAT,
)
def add_pic_to_database(name, probabilitythreshold, dbpath):
    db = Database()

    img = cam.take_picture()

    fingerprints = mw.compute_fingerprints(
        img, mw.feed_mtcnn(img, threshold=probabilitythreshold)
    )

    db.add_multi(name, list(fingerprints))
    db.save()


@cli.command()
@ck.argument("filename")
@ck.option("-t", "--threshold", type=ck.FLOAT, default=2.0)
@ck.option("-p", "--probabilitythreshold", type=ck.FLOAT, default=0.8)
@ck.option("-d", "--dbpath", help="Path to database")
def find_faces(filename, threshold, probabilitythreshold, dbpath):
    """Command to find and identify faces in an image, label them with boxes and names from the database, and 
	display a final image with the aforementioned names and boxes. It labels unknown faces as "Unknown."""
    img = inp.import_image(Path(filename))

    boxes = mw.feed_mtcnn(img, threshold=probabilitythreshold)

    fingerprints = mw.compute_fingerprints(img, boxes)

    names = dm.determine_matches(fingerprints, threshold=threshold)

    imd.display_image(img, boxes, names)


@cli.command()
@ck.argument("foldername")
@ck.option("-t", "--threshold", type=ck.FLOAT, default=1)
@ck.option("-m", "--maxiterations", type=ck.INT, default=200)
@ck.option("-w", "--weightededges", type=ck.BOOL, default=True)
def whispers(foldername, threshold, maxiterations, weightededges):
    """Runs the whispers algorithm on a folder of images and graphs the final graph once it is complete"""
    graph = whisp.create_graph(foldername, threshold=threshold)

    adjacency_matrix = whisp.whispers(
        graph, max_iterations=maxiterations, weighted_edges=weightededges
    )

    nd.plot_graph(graph, adjacency_matrix)
