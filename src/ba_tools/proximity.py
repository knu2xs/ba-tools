from concurrent import futures
import math
import multiprocessing
import os
import tempfile
import uuid

from arcgis.network.analysis import find_closest_facilities
from arcgis.features import GeoAccessor
from arcgis.geometry import Geometry
import arcpy
import pandas as pd

from . import utils
from ._data import data

# location to store temp files if necessary
csv_file_prefix = 'temp_closest'
temp_file_root = os.path.join(tempfile.gettempdir(), csv_file_prefix)

# ensure previous runs do not interfere
arcpy.env.overwriteOutput = True


def weighted_polygon_centroid(poly_df, wgt_df, poly_id_fld='ID'):
    # create a spatial index for both spatially enabled dataframes to speed up the join
    if poly_df.spatial._sindex is None:
        poly_df.spatial.sindex()
    if wgt_df.spatial._sindex is None:
        wgt_df.spatial.sindex()

    # perform a spatial join between the points and the containing polygons
    join_df = wgt_df.spatial.join(poly_df)

    # extract the respective coordinates out of the geometry
    join_df['x'] = join_df['SHAPE'].apply(lambda geom: geom.centroid[0])
    join_df['y'] = join_df['SHAPE'].apply(lambda geom: geom.centroid[1])

    # get the id field following the join
    join_id_fld = poly_id_fld if poly_id_fld in join_df.columns else f'{poly_id_fld}_right'

    # calcuate the weighted centroid coordinates for each polygon and create geometries using these
    mean_df = join_df[[join_id_fld, 'x', 'y']].groupby(join_id_fld).mean()
    mean_df['SHAPE'] = mean_df.apply(
        lambda row: Geometry({'x': row['x'], 'y': row['y'], 'spatialReference': {'wkid': 4326}}), axis=1)

    # clean up the columns to a standard output schema
    mean_df.reset_index(inplace=True)
    mean_df = mean_df[[join_id_fld, 'SHAPE']].copy()
    mean_df.spatial.set_geometry('SHAPE')
    mean_df.columns = ['ID', 'SHAPE']

    return mean_df


def prep_sdf_for_nearest(df, id_fld, weighting_points=None):
    """
    Given an input Spatially Enabled Dataframe, prepare it to work well with the nearest solver.
    :param df: Spatially Enabled Dataframe with really any geometry.
    :param id_fld: Field uniquely identifying each of location to be used for routing to nearest.
    :param weighting_points: Spatially Enabled Dataframe of points for calculating weighted centroids.
    :return: Spatially Enabled Dataframe of points with correct columns for routing to nearest.
    """
    # par down the input dataframe to just the columns needed
    df = df[[id_fld, 'SHAPE']].copy()

    # rename the columns to follow the schema needed for routing
    df.columns = ['ID', 'SHAPE']

    # if the geometry is polygons and there is a weighting points spatially enabled dataframe
    if df.spatial.geometry_type == ['polygon'] and weighting_points is not None:

        # calculate the weighted centroid for the polygons
        df = weighted_polygon_centroid(df, weighting_points)

    # otherwise, if the geometry is not points, we still need points, so just get the geometric centroids
    # TODO: Account for polygons NOT always being in WGS 84
    elif df.spatial.geometry_type != ['point'] and weighting_points is None:
        df['SHAPE'] = df['SHAPE'].apply(
            lambda geom: Geometry({'x': geom.centroid[0], 'y': geom.centroid[1], 'spatialReference': {'wkid': 4326}}))

    # add a second column for the ID as Name
    df['Name'] = df['ID']

    # ensure the geometry is correctly being recognized
    df.spatial.set_geometry('SHAPE')

    # set the order of the columns and return
    return df[['ID', 'Name', 'SHAPE']].copy()


def prep_feature_service_for_nearest(item_id, id_fld, gis):
    """
    Using a feature service as the input, format the schema for closest routing and return as a Spatially Enabled
        Dataframe.
    :param item_id: The item ID in the Web GIS the feature service can be found at.
    :param id_fld: The ID field uniquely identifying each resource.
    :param gis: The Web GIS object instance to use for connecting to get the data.
    :return: Spatially Enabled Dataframe with correct schema for closest network analysis.
    """
    df = gis.content.get(item_id).layers[0].query(out_sr=4326, as_df=True)
    return prep_sdf_for_nearest(df, id_fld)


def _get_max_near_dist_arcpy(origin_lyr):
    """
    Get the maximum geodesic distance between stores.
    """
    # create a location for temporary data
    temp_table = r'in_memory\near_table_{}'.format(uuid.uuid4().hex)

    # if only one location, cannot generate a near table, and default to 120 miles
    if int(arcpy.management.GetCount(origin_lyr)[0]) <= 1:
        max_near_dist = 120 * 1609.34

    else:
        # use arcpy to get a table of all distances between stores
        near_tbl = arcpy.analysis.GenerateNearTable(
            in_features=origin_lyr,
            near_features=origin_lyr,
            out_table=temp_table,
            method="GEODESIC"
        )[0]

        # get the maximum near distance, which will be in meters
        meters = max([row[0] for row in arcpy.da.SearchCursor(near_tbl, 'NEAR_DIST')])

        # remove the temporty table to ensure not stuff lying around and consuming RAM
        arcpy.management.Delete(temp_table)

        # get the maximum near distance (in meters)
        max_near_dist = meters * 0.00062137

    return max_near_dist


def _get_closest_df_rest(origin_df, dest_df, dest_count, gis, max_dist=None):
    """
    Succinct function wrapping find_closest_facilities with a little ability to handle network and server hiccups
    :param origin_df: Origin points Spatially Enabled Dataframe
    :param dest_df: Destination points Spatially Enabled Dataframe
    :param dest_count: Destination points Spatially Enabled Dataframe
    :param gis: ArcGIS Web GIS object instance with networking configured.
    :param max_dist: Maximum nearest routing distance in miles.
    :return: Spatially Enabled Dataframe of solved closest facility routes.
    """
    attempt_count = 0
    max_attempts = 10
    while attempt_count < max_attempts:
        try:
            closest_result = find_closest_facilities(
                incidents=origin_df.spatial.to_featureset().to_json,
                facilities=dest_df.spatial.to_featureset().to_json,
                measurement_units='Miles',
                number_of_facilities_to_find=dest_count,
                cutoff=max_dist,
                travel_direction='Incident to Facility',
                use_hierarchy=False,
                restrictions="['Avoid Private Roads', 'Driving an Automobile', 'Roads Under Construction Prohibited', "
                             "'Avoid Gates', 'Avoid Express Lanes', 'Avoid Carpool Roads']",
                impedance='Travel Distance',
                gis=gis
            )
            return closest_result.output_routes.sdf

        except:
            attempt_count = attempt_count + 1


def _get_closest_df_arcpy(origin_df, dest_df, dest_count, network_dataset, max_dist=None):
    """
    Succinct function wrapping find_closest_facilities with a little ability to handle network and server hiccups
    :param origin_df: Origin points Spatially Enabled Dataframe
    :param dest_df: Destination points Spatially Enabled Dataframe
    :param dest_count: Destination points Spatially Enabled Dataframe
    :param network_dataset: Path to ArcGIS Network dataset.
    :param max_dist: Maximum nearest routing distance in miles.
    :return: Spatially Enabled Dataframe of solved closest facility routes.
    """
    # get the mode of travel from the network dataset - rural so gravel roads are fair game
    nd_lyr = arcpy.nax.MakeNetworkDatasetLayer(network_dataset)[0]
    trvl_mode_dict = arcpy.nax.GetTravelModes(nd_lyr)
    trvl_mode = trvl_mode_dict['Rural Driving Time']

    # create the closest solver object instance
    # https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/closestfacility.htm
    closest_solver = arcpy.nax.ClosestFacility(network_dataset)

    # set parameters for the closest solver
    closest_solver.travelMode = trvl_mode
    closest_solver.travelDirection = arcpy.nax.TravelDirection.ToFacility
    # TODO: How to set this to distance?
    closest_solver.timeUnits = arcpy.nax.TimeUnits.Minutes
    closest_solver.distanceUnits = arcpy.nax.DistanceUnits.Miles
    closest_solver.defaultTargetFacilityCount = dest_count
    closest_solver.routeShapeType = arcpy.nax.RouteShapeType.TrueShapeWithMeasures
    closest_solver.searchTolerance = 5000
    closest_solver.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

    # since maximum distance is optional, well, make it optional
    if max_dist:
        closest_solver.defaultImpedanceCutoff = max_dist

    # load the origin and destination feature data frames into memory and load into the solver object instance
    origin_fc = origin_df.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'origin'))
    closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Incidents, origin_fc)

    dest_fc = dest_df.spatial.to_featureclass(os.path.join(arcpy.env.scratchGDB, 'dest'))
    closest_solver.load(arcpy.nax.ClosestFacilityInputDataType.Facilities, dest_fc)

    # run the solve, and get comfortable
    closest_result = closest_solver.solve()

    # export the results to a spatially enabled data frame
    route_fc = 'in_memory/routes'
    closest_result.export(arcpy.nax.ClosestFacilityOutputDataType.Routes, route_fc)
    route_oid_col = arcpy.Describe(route_fc).OIDFieldName
    closest_df = GeoAccessor.from_featureclass(route_fc)
    if route_oid_col:
        closest_df.drop(columns=[route_oid_col], inplace=True)
    arcpy.management.Delete(route_fc)

    # get rid of the extra empty columns the local network solve adds
    closest_df.dropna(axis=1, how='all', inplace=True)

    # populate the origin and destination fields so the schema matches what online solve returns
    name_srs = closest_df.Name.str.split(' - ')
    closest_df['IncidentID'] = name_srs.apply(lambda val: val[0])
    closest_df['FacilityID'] = name_srs.apply(lambda val: val[1])

    return closest_df


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


def _get_closest_csv_rest(origin_df, dest_df, dest_count, gis, max_dist=None):
    """
    Enables batch processing of get closest by saving iterative results to a temp csv file to avoid memory overruns.
    :param origin_df: Origin points Spatially Enabled Dataframe
    :param dest_df: Destination points Spatially Enabled Dataframe
    :param dest_count: Destination points Spatially Enabled Dataframe
    :param gis: ArcGIS Web GIS object instance with networking configured.
    :param max_dist: Maximum nearest routing distance in miles.
    :return: String path to CSV of solved closest facility routes.
    """
    out_csv_path = f'{temp_file_root}_{uuid.uuid4().hex}.csv'
    closest_df = _get_closest_df_rest(origin_df, dest_df, dest_count, gis, max_dist)
    closest_df.to_csv(out_csv_path)
    return out_csv_path


def reformat_closest_result_dataframe(closest_df):
    """
    Reformat the schema, dropping unneeded coluns and renaming those kept to be more in line with this workflow.
    :param closest_df: Dataframe of the raw output routes from the find closest analysis.
    :return: Spatially Enabled Dataframe reformatted.
    """
    # create a list of columns containing proximity metrics
    proximity_src_cols = [col for col in closest_df.columns if col.startswith('Total_')]

    # if both miles and kilometers, drop miles, and keep kilometers
    miles_lst = [col for col in proximity_src_cols if 'miles' in col.lower()]
    kilometers_lst = [col for col in proximity_src_cols if 'kilometers' in col.lower()]
    if len(miles_lst) and len(kilometers_lst):
        proximity_src_cols = [col for col in proximity_src_cols if col != miles_lst[0]]

    # calculate side of street columns
    closest_df['proximity_side_street_right'] = (closest_df['FacilityCurbApproach'] == 1).astype('int64')
    closest_df['proximity_side_street_left'] = (closest_df['FacilityCurbApproach'] == 2).astype('int64')
    side_cols = ['proximity_side_street_left', 'proximity_side_street_right']

    # filter the dataframe to just the columns we need
    src_cols = ['IncidentID', 'FacilityRank', 'FacilityID'] + proximity_src_cols + side_cols + ['SHAPE']
    closest_df = closest_df[src_cols].copy()

    # replace total in proximity columns for naming convention
    closest_df.columns = [col.lower().replace('total', 'proximity') if col.startswith('Total_') else col
                          for col in closest_df.columns]

    # rename the columns for the naming convention
    rename_dict = {'IncidentID': 'origin_id', 'FacilityRank': 'destination_rank', 'FacilityID': 'destination_id'}
    closest_df = closest_df.rename(columns=rename_dict)

    return closest_df


def explode_closest_rank_dataframe(closest_df, origin_id_col='origin_id', rank_col='destination_rank',
                                   dest_id_col='destination_id'):
    """
    Effectively explode out or pivot the data so there is only a single record for each origin.
    :param closest_df: Spatially Enabled Dataframe reformatted from the raw output of find nearest.
    :param origin_id_col: Column uniquely identifying each origin - default 'origin_id'
    :param rank_col: Column identifying the rank of each destination - default 'destination_rank'
    :param dest_id_col: Column uniquely identifying each destination - default 'destination_id'
    :return: Dataframe with a single row for each origin with multiple destination metrics for each.
    """
    # create a dataframe to start working with comprised of only the unique origins to start with
    origin_dest_df = pd.DataFrame(closest_df[origin_id_col].unique(), columns=[origin_id_col])

    # get a list of the proximity columns
    proximity_cols = [col for col in closest_df.columns if col.startswith('proximity_')]

    # iterate the closest destination ranking
    for rank_val in closest_df[rank_col].unique():

        # filter the dataframe to just the records with this destination ranking
        rank_df = closest_df[closest_df[rank_col] == rank_val]

        # create a temporary dataframe to begin building the columns onto
        df_temp = rank_df[origin_id_col].to_frame()

        # iterate the relevant columns
        for col in [dest_id_col] + proximity_cols:
            # create a new column name from the unique value and the original row name
            new_name = f'{col}_{rank_val:02d}'

            # filter the data in the column with the unique value
            df_temp[new_name] = rank_df[col].values

        # set the index to the origin id for joining
        df_temp.set_index(origin_id_col, inplace=True)

        # join the temporary dataframe to the master
        origin_dest_df = origin_dest_df.join(df_temp, on=origin_id_col)

    return origin_dest_df


def get_closest_solution(origins, origin_id_fld, destinations, dest_id_fld, network_dataset=None, destination_count=4,
                         gis=None):
    """
    Create a closest destination dataframe using origin and destination Spatially Enabled Dataframes, but keep
        each origin and destination still in a discrete row instead of collapsing to a single row per origin. The main
        reason to use this is if needing the geometry for visualization.
    :param origins: Spatially Enabled Dataframe | String path to Feature Class | String url to Feature Service |
        String Web GIS Item ID
        Origins in one of the supported input formats.
    :param origin_id_fld: Column in the origin points Spatially Enabled Dataframe uniquely identifying each feature
    :param destinations: Spatially Enabled Dataframe | String path to Feature Class | String url to Feature Service |
        String Web GIS Item ID
        Destination points in one of the supported input formats.
    :param dest_id_fld: Column in the destination points Spatially Enabled Dataframe uniquely identifying each feature
    :param network_dataset: Path to ArcGIS Network dataset.
    :param destination_count: Integer number of destinations to search for from every origin point.
    :param gis: ArcGIS Web GIS object instance with networking configured.
    :return: Spatially Enabled Dataframe with a row for each origin id, and metrics for each nth destinations.
    """
    # check to environment against inputs to determine if networking locally or remotely
    if gis is not None and network_dataset is not None:
        raise Exception('You can either specify a GIS object instance OR a Network Dataset, but not both.')

    # ensure the inputs are a spatially enabled dataframe
    origin_df = utils.get_dataframe(origins, gis)
    dest_df = utils.get_dataframe(destinations, gis)

    # ensure the dataframes are in the right schema and have the right geometry
    origin_df = prep_sdf_for_nearest(origin_df, origin_id_fld)
    dest_df = prep_sdf_for_nearest(dest_df, dest_id_fld)

    # create an environment object instance for checking settings later
    env = utils.Environment(gis)

    # yes, the undocumented way to use a REST endpoint happens here...
    if gis is not None:

        # get the limitations on the networking rest endpoint, and scale the analysis based on this
        max_records = gis._con.get(gis.properties.helperServices.asyncClosestFacility.url.rpartition('/')[0])['maximumRecords']
        max_origin_cnt = math.floor(max_records / destination_count)

        # if necessary, batch the analysis based on size of the input data, and the number of destinations per origin
        if len(origin_df.index) > max_origin_cnt:

            # process each batch, and save the results to a temp file in the temp directory
            closest_csv_list = [_get_closest_csv_rest(origin_df.iloc[idx:idx + max_origin_cnt], dest_df, destination_count, gis)
                                for idx in range(0, len(origin_df.index), max_origin_cnt)]

            # load all the temporary files into dataframes and combine them into a single dataframe
            closest_df = pd.concat([pd.read_csv(closest_csv) for closest_csv in closest_csv_list])

            # clean up the temp files
            for csv_file in closest_csv_list:
                os.remove(csv_file)

        else:
            closest_df = _get_closest_df_rest(origin_df, dest_df, destination_count, gis)

        # reformat the results to be a single row for each origin
        closest_df = reformat_closest_result_dataframe(closest_df)

    else:

        # check to make sure network analyst is available
        if 'Network' in env.arcpy_extensions:
            env.arcpy_checkout_extension('Network')
        else:
            raise Exception('To perform network routing locally you must have access to the ArcGIS Network Analyst '
                            'extension. It appears this extension is either not installed or not licensed.')

        # try to get a network to work with if one is not provided
        if network_dataset is None:

            # if the local usa data is installed, use it, but if not, we don't have anything to work with
            if data.usa_network_dataset:
                network_dataset = data.usa_network_dataset
            else:
                raise Exception('You must either have a ')

        # run the closest analysis locally
        closest_df = _get_closest_df_arcpy(origin_df, dest_df, destination_count, network_dataset)

        # reformat the results to be a single row for each origin
        closest_df = reformat_closest_result_dataframe(closest_df)

    return closest_df


def closest_dataframe_from_origins_destinations(origins, origin_id_fld, destinations, dest_id_fld, gis=None,
                                                network_dataset=None, destination_count=4):
    """
    Create a closest destination dataframe using origin and destination Spatially Enabled Dataframes.
    :param origins: Spatially Enabled Dataframe | String path to Feature Class | String url to Feature Service |
        String Web GIS Item ID
        Origins in one of the supported input formats.
    :param origin_id_fld: Column in the origin points Spatially Enabled Dataframe uniquely identifying each feature
    :param destinations: Spatially Enabled Dataframe | String path to Feature Class | String url to Feature Service |
        String Web GIS Item ID
        Destination points in one of the supported input formats.
    :param dest_id_fld: Column in the destination points Spatially Enabled Dataframe uniquely identifying each feature
    :param gis: ArcGIS Web GIS object instance with networking configured.
    :param network_dataset: Path to ArcGIS Network dataset.
    :param destination_count: Integer number of destinations to search for from every origin point.
    :return: Spatially Enabled Dataframe with a row for each origin id, and metrics for each nth destinations.
    """
    # get a closest dataframe with all the origin and destination pairs in a discrete row
    closest_df = get_closest_solution(origins, origin_id_fld, destinations, dest_id_fld, gis=gis,
                                      network_dataset=network_dataset, destination_count=destination_count)

    # collapse the solutions to a single record for each origin location
    origin_dest_df = explode_closest_rank_dataframe(closest_df)

    return origin_dest_df
