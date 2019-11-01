"""
Methods to analyze hypotehtical scenarios.
"""

import arcgis
import arcpy
from arcgis.geometry import Point
import logging
import pandas as pd
import pathlib
from tempfile import gettempdir

from ._data import data
from . import proximity
from .utils import get_dataframe, get_logger
from .enrich import enrich_all


def _get_min_uid(df, uid_field, start_value=None):
    match = False
    idx = start_value if start_value else 1
    while match is False:
        if idx not in df[uid_field].astype('int64').values:
            uid = idx
            match = True
        elif idx >= df[uid_field].astype('int64').max():
            idx = idx + 1000
        else:
            idx = idx + 1
    return uid


def get_master_dataframe(origin_geography_layer: arcpy._mp.Layer, origin_id_field: str,
                         brand_location_layer: arcpy._mp.Layer, brand_id_field: str,
                         competitor_location_layer: arcpy._mp.Layer, competitor_id_field: str,
                         destination_count: int = 6, overwrite_intermediate: bool = False,
                         logger: logging.Logger = None):
    """
    Build the master dataframe used for initial model development using a combination of geographic customer origin
        geographies, brand locations, and competitor locations.
    :param origin_geography_layer: Layer of origin geographies where people are coming from - ideally US Census
        Block Groups.
    :param origin_id_field: String field name uniquely identifying the origin geographies.
    :param brand_location_layer: Point location layer where the brand locations are at.
    :param brand_id_field: String field name uniquely identifying the brand locations.
    :param competitor_location_layer: Point location layer where the competitor locations are at.
    :param competitor_id_field: String field name uniquely identifying the competitor locations.
    :param destination_count: Optional integer count of the number of locations to find for each origin geography.
        The default is six.
    :param overwrite_intermediate: Optional boolean indicating if this analysis should overwrite previous runs. The
        default is False indicating to use previous runs if previous attempts were unsuccessful.
    :param logger: Optional Logger object instance with details for saving results of attempting to build data.
    :return: Pandas dataframe with all the data assembled and ready for modeling.
    """
    # set up logging
    if logger is None:
        logger = get_logger('INFO')

    # get a temporary directory, the standard one, to work with
    temp_dir = pathlib.Path(gettempdir())

    # set paths for where to save intermediate data results
    enrich_all_out = temp_dir / 'origin_enrich_all.csv'
    nearest_brand_out = temp_dir / 'nearest_brand.csv'
    nearest_comp_out = temp_dir / 'nearest_competition.csv'

    # if starting from scratch, clean everything out
    if overwrite_intermediate:
        for out_file in [enrich_all_out, nearest_brand_out, nearest_comp_out]:
            if out_file.exists():
                out_file.unlink()

    enrich_df, nearest_brand_df, nearest_comp_df = None, None, None

    # enrich all contributing origin geographies with all available demographics
    if not enrich_all_out.exists() or overwrite_intermediate:
        try:
            logger.info(f'Starting to enrich {origin_geography_layer}.')
            enrich_df = enrich_all(origin_geography_layer, id_field=origin_id_field)
            enrich_df.rename({origin_id_field: 'origin_id'}, axis=1, inplace=True)
            enrich_df.set_index('origin_id', drop=True, inplace=True)
            enrich_df.to_csv(str(enrich_all_out))
            logger.info(
                f'Successfully enriched origin geographies. The output is located at {str(enrich_all_out)}.')

        except Exception as e:
            logger.error(f'Failed to enrich {origin_geography_layer}.\n{e}')

    else:
        enrich_df = pd.read_csv(enrich_all_out)
        logger.info(f'Enriched origin geographies already exist at {str(enrich_all_out)}.')

    # create a nearest table for all store locations
    if not nearest_brand_out.exists() or overwrite_intermediate:
        try:
            logger.info('Starting to find closest store locations.')
            nearest_brand_df = proximity.closest_dataframe_from_origins_destinations(
                origin_geography_layer, origin_id_field, brand_location_layer, brand_id_field,
                network_dataset=data.usa_network_dataset, destination_count=destination_count
            )
            nearest_brand_df.set_index('origin_id', drop=True, inplace=True)
            nearest_brand_df.to_csv(str(nearest_brand_out))
            logger.info('Successfully solved closest store locations.')

        except Exception as e:
            logger.error(f'Failed to solve closest stores.\n{e}')

    else:
        nearest_brand_df = pd.read_csv(nearest_brand_out, index_col=0)
        logger.info(f'Closest store solution already exists at {str(nearest_brand_out)}.')

    # create a nearest table for all competition locations
    if not nearest_comp_out.exists():
        try:
            logger.info('Starting to find closest competition locations')
            nearest_comp_df = proximity.closest_dataframe_from_origins_destinations(
                origin_geography_layer, origin_id_field, competitor_location_layer,
                competitor_id_field, network_dataset=data.usa_network_dataset, destination_count=destination_count
            )
            nearest_comp_df.set_index('origin_id', drop=True, inplace=True)
            nearest_comp_df.columns = [c.replace('proximity', 'proximity_competition') for c in
                                       nearest_comp_df.columns]
            nearest_comp_df.columns = [c.replace('destination', 'destination_competition') for c in
                                       nearest_comp_df.columns]
            nearest_comp_df.to_csv(str(nearest_comp_out))
            logger.info('Successfully solved closest competition locations.')

        except Exception as e:
            logger.error(f'Failed to solve closest competition.\n{e}')

    else:
        nearest_comp_df = pd.read_csv(nearest_comp_out)
        logger.info(f'Closest competition solution already exists at {str(nearest_comp_out)}')

    # if we made it this far, and all three dataframes were successfully created, assemble into an output dataframe
    if enrich_df is None or nearest_brand_df is None or nearest_comp_df is None:
        raise Exception('Could not create all three output results. Please view logs to see more.')
    else:
        master_df = enrich_df.join(nearest_brand_df).join(nearest_comp_df)

        return master_df


def get_master_csv(origin_geography_layer: arcpy._mp.Layer, origin_id_field: str,
                   brand_location_layer: arcpy._mp.Layer, brand_id_field: str,
                   competitor_location_layer: arcpy._mp.Layer, competitor_id_field: str,
                   output_csv_file: [str, pathlib.Path], destination_count: int = 6,
                   overwrite_intermediate: bool = False, logger: logging.Logger = None):
    """
    Build the master dataframe used for initial model development and save as a CSV using a combination of
        geographic customer origin geographies, brand locations, and competitor locations.
    :param origin_geography_layer: Layer of origin geographies where people are coming from - ideally US Census
        Block Groups.
    :param origin_id_field: String field name uniquely identifying the origin geographies.
    :param brand_location_layer: Point location layer where the brand locations are at.
    :param brand_id_field: String field name uniquely identifying the brand locations.
    :param competitor_location_layer: Point location layer where the competitor locations are at.
    :param competitor_id_field: String field name uniquely identifying the competitor locations.
    :param output_csv_file: Path to output CSV file where the prepped data will be saved.
    :param destination_count: Optional integer count of the number of locations to find for each origin geography.
        The default is six.
    :param overwrite_intermediate: Optional boolean indicating if this analysis should overwrite previous runs. The
        default is False indicating to use previous runs if previous attempts were unsuccessful.
    :param logger: Optional Logger object instance with details for saving results of attempting to build data.
    :return: Pandas dataframe with all the data assembled and ready for modeling.
    """
    master_df = get_master_dataframe(origin_geography_layer, origin_id_field, brand_location_layer, brand_id_field,
                                     competitor_location_layer, competitor_id_field, destination_count,
                                     overwrite_intermediate, logger)

    master_df.to_csv(output_csv_file)

    if not isinstance(pathlib.Path, output_csv_file):
        output_csv_file = pathlib.Path(output_csv_file)
    return output_csv_file


def get_add_new_closest_dataframe(origins: [str, pd.DataFrame], origin_id_field: str, destinations: [str, pd.DataFrame],
                                  destination_id_field: str, closest_table: [str, pd.DataFrame], new_destination: Point,
                                  gis: arcgis.gis.GIS = None, origin_weighting_points: [str, pd.DataFrame] = None
                                  ) -> pd.DataFrame:
    """
    Calculate the impact of a location being added to the retail landscape.
    :param origins: Polygons in a Spatially Enabled Dataframe or string path to Feature Class delineating starting
        locations for closest analysis.
    :param origin_id_field: Field or column name used to uniquely identify each origin.
    :param destinations: Spatially Enabled Dataframe or string path to Feature Class containing all destinations.
    :param destination_id_field: Field or column name used to uniquely identify each destination location.
    :param closest_table: Path to CSV, table, or Dataframe containing solution for nearest locations.
    :param new_destination: Geometry of new location being added to the retail landscape.
    :param origin_weighting_points: Points potentially used to calculate a centroid based on population density
        represented by the weighting points instead of simply the geometric centroid.
    :return: Data frame with rebalanced closest table only for affected origins.
    """
    # read in the existing closest table solution
    closest_orig_df = pd.read_csv(closest_table)

    # get a list of the destination columns from the existing closest table
    dest_cols = [col for col in closest_orig_df.columns if col.startswith('destination_id')]

    # get a count of the nth number of locations solved for
    dest_count = len(dest_cols)

    # load the original origins into a dataframe and format it for analysis
    origin_df = get_dataframe(origins)
    origin_df = proximity.prep_sdf_for_nearest(origin_df, origin_id_field, origin_weighting_points)

    # load the original destinations into a dataframe and format it for analysis
    dest_df = get_dataframe(destinations)
    dest_df = proximity.prep_sdf_for_nearest(dest_df, destination_id_field)

    # create new destination dataframe for analysis
    new_id = _get_min_uid(origin_df, 'ID')  # creates lowest numbered id available, or 1000 higher than top value
    new_df = pd.DataFrame([[new_id, new_id, new_destination]], columns=['ID', 'Name', 'SHAPE'])
    new_df.spatial.set_geometry('SHAPE')

    # if a GIS is provided, use online resources
    if gis is not None:
        # get the nth closest destination locations to the new destination location
        closest_dest_df = proximity.get_closest_solution(new_df, 'ID', dest_df, 'ID', gis=gis,
                                                         destination_count=dest_count)

    # otherwise, use local solver
    else:

        # get the nth closest destination locations to the new destination location
        closest_dest_df = proximity.get_closest_solution(new_df, 'ID', dest_df, 'ID',
                                                         network_dataset=data.usa_network_dataset,
                                                         destination_count=dest_count)

    # get the destination ids of the existing nth closest destinations
    dest_subset_ids = closest_dest_df['destination_id'].values

    # by cross referencing from the destination ids, get the origin ids allocated to the exiting locations
    subset_origin_ids = pd.concat([closest_orig_df[closest_orig_df[dest_col].isin(dest_subset_ids)]['origin_id']
                                   for dest_col in dest_cols]).unique()

    # get a subset dataframe of the origins allocated to the closest nth locations
    subset_origin_df = origin_df[origin_df['ID'].astype('int64').isin(subset_origin_ids)].copy()

    # add the new location to the destination dataframe
    dest_analysis_df = pd.concat([dest_df, new_df], sort=False)
    dest_analysis_df.spatial.set_geometry('SHAPE')
    dest_analysis_df.reset_index(inplace=True, drop=True)

    # if a GIS is provided, use online resources to solve for the closest destination to the affected area
    if gis is not None:

        # solve for the closest destination to the affected area
        closest_subset_df = proximity.closest_dataframe_from_origins_destinations(subset_origin_df, 'ID',
                                                                                  dest_analysis_df, 'ID',
                                                                                  gis=gis,
                                                                                  destination_count=dest_count)

    # otherwise, use local resources
    else:

        # solve for the closest destination to the affected area
        closest_subset_df = proximity.closest_dataframe_from_origins_destinations(subset_origin_df, 'ID',
                                                                                  dest_analysis_df, 'ID',
                                                                                  network_dataset=data.usa_network_dataset,
                                                                                  destination_count=dest_count)

    return closest_subset_df


def get_remove_existing_closest_dataframe(origins: [str, pd.DataFrame], origin_id_field: str,
                                          destinations: [str, pd.DataFrame],
                                          destination_id_field: str, closest_table: [str, pd.DataFrame],
                                          remove_destination_id: str,
                                          origin_weighting_points: [str, pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate the impact of a location being removed from the retail landscape.
    :param origins: Polygons in a Spatially Enabled Dataframe or string path to Feature Class delineating starting
        locations for closest analysis.
    :param origin_id_field: Field or column name used to uniquely identify each origin.
    :param destinations: Spatially Enabled Dataframe or string path to Feature Class containing all destinations.
    :param destination_id_field: Field or column name used to uniquely identify each destination location.
    :param closest_table: Path to CSV, table, or Dataframe containing solution for nearest locations.
    :param remove_destination_id: Unique ID of location being removed from the retail landscape.
    :param origin_weighting_points: Points potentially used to calculate a centroid based on population density
        represented by the weighting points instead of simply the geometric centroid.
    :return: Data frame with rebalanced closest table only for affected origins.
    """
    # read in the existing closest table solution
    closest_orig_df = pd.read_csv(closest_table)

    # get a list of the destination columns from the existing closest table
    dest_cols = [col for col in closest_orig_df.columns if col.startswith('destination_id')]

    # get a count of the nth number of locations solved for
    dest_count = len(dest_cols)

    # load the original origins into a dataframe and format it for analysis
    origin_df = get_dataframe(origins)
    origin_df = proximity.prep_sdf_for_nearest(origin_df, origin_id_field, origin_weighting_points)

    # load the original destinations into a dataframe and format it for analysis
    dest_df = get_dataframe(destinations)
    dest_df = proximity.prep_sdf_for_nearest(dest_df, destination_id_field)

    # extract the location from the destinations to be removed and put it in a separate dataframe
    new_df = dest_df[dest_df['ID'] == str(remove_destination_id)].copy()

    # remove the location from the destinations
    dest_df = dest_df[dest_df['ID'] != str(remove_destination_id)].copy()

    # get the nth closest destination locations to the new destination location
    closest_dest_df = proximity.get_closest_solution(new_df, 'ID', dest_df, 'ID',
                                                     network_dataset=data.usa_network_dataset,
                                                     destination_count=dest_count)

    # get the destination ids of the existing nth closest destinations
    dest_subset_ids = closest_dest_df['destination_id'].values

    # by cross referencing from the destination ids, get the origin ids allocated to the exiting locations
    subset_origin_ids = pd.concat([closest_orig_df[closest_orig_df[dest_col].isin(dest_subset_ids)]['origin_id']
                                   for dest_col in dest_cols]).unique()

    # get a subset dataframe of the origins allocated to the closest nth locations
    subset_origin_df = origin_df[origin_df['ID'].astype('int64').isin(subset_origin_ids)].copy()

    # solve for the closest destination to the affected area
    closest_subset_df = proximity.closest_dataframe_from_origins_destinations(subset_origin_df, 'ID', dest_df,
                                                                              'ID',
                                                                              network_dataset=data.usa_network_dataset,
                                                                              destination_count=dest_count)

    return closest_subset_df
