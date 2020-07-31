from concurrent import futures
import math
import multiprocessing

from . import utils
from .proximity import _get_closest_df_arcpy


def _get_closest_arcpy_multithreaded(origin_df, dest_df, dest_count, network_dataset, max_dist=None):
    """
    Multithread and speed up the process of using local networks for analysis.
    :param origin_df: Origin points Spatially Enabled Dataframe
    :param dest_df: Destination points Spatially Enabled Dataframe
    :param dest_count: Destination points Spatially Enabled Dataframe
    :param network_dataset: Path to ArcGIS Network dataset.
    :param max_dist: Maximum nearest routing distance in miles.
    :return: Spatially Enabled Dataframe of solved closest facility routes.
    """
    # set the worker count to one less than the number of processors available
    workers = (multiprocessing.cpu_count() - 1)

    # if there are less origins than available workers, reduce the worker count to the number of origins
    if len(origin_df.index) < workers:
        workers = len(origin_df.index)

    # set the batch size based on the number of workers available
    batch_size = math.floor(len(origin_df.index) / workers)

    # get a list of index tuples for slicing
    batch_idx_lst = utils.blow_chunks(origin_df.index, batch_size)

    # helper for iteratively invoking closest arcpy
    def _multiprocess_closest_arcpy(idx):
        chunk_origin_df = origin_df[idx.start: idx.stop]
        return _get_closest_df_arcpy(chunk_origin_df, dest_df, dest_count, network_dataset)

    # split apart job across cores
    with futures.ProcessPoolExecutor(max_workers=workers) as executors:
        results = executors.map(_multiprocess_closest_arcpy, batch_idx_lst)
        out_df_lst = []
        for result in results:
            out_df_lst.append(result)
        return out_df_lst
