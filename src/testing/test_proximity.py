import os
from pathlib import Path
import sys

from arcgis import GeoAccessor
from arcgis.geometry import Geometry
import arcpy
import pandas as pd
import pytest

# facilitate local resource imports
project_dir_str = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
src_dir_str = project_dir_str
sys.path.insert(0, str(src_dir_str))
from ba_tools import data, proximity, utils

data_dir = Path(__file__).parent / 'test_data'
gdb = data_dir / 'test_data.gdb'

block_groups = gdb / 'block_groups'
block_group_id_field = 'ID'

brand_locs = gdb / 'locations_brand'
brand_id_field = 'LOCNUM'

comp_locs = gdb / 'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir / 'origin_demographics.csv'
nearest_brand_csv = data_dir / 'nearest_locations.csv'

scratch_dir = data_dir / 'scratch'

closest_parquet = data_dir / 'closest_test.parquet'


@pytest.fixture
def block_group_points_df():
    df = utils.get_dataframe(block_groups).drop(columns='OBJECTID')
    df.SHAPE = df.SHAPE.apply(lambda geom: Geometry({"x": geom.centroid[0], "y": geom.centroid[1],
                                                     "spatialReference": df.spatial.sr}))
    df.spatial.set_geometry('SHAPE')
    return df


@pytest.fixture
def brand_df():
    return utils.get_dataframe(brand_locs).drop(columns='OBJECTID')


@pytest.fixture
def closest_df_test():
    df = pd.read_parquet('./test_data/closest_test.parquet')
    df.SHAPE = df.SHAPE.apply(lambda geom: Geometry(eval(geom)))
    df.spatial.set_geometry('SHAPE')
    return df


def test_get_closest_df_arcpy(block_group_points_df, brand_df, closest_df_test):
    closest_df = proximity._get_closest_df_arcpy(block_group_points_df, brand_df, 6, data.usa_network_dataset)
    cols = ['FacilityRank', 'Name', 'IncidentCurbApproach', 'FacilityCurbApproach', 'IncidentID', 'Total_Miles',
            'Total_Kilometers', 'Total_Minutes', 'FacilityOID', 'IncidentOID', 'SHAPE', 'FacilityID']
    assert (list(closest_df.columns) == cols)

# def test_get_closest_df_arcpy_multithreaded(block_group_points_df, brand_df, closest_df_test):
#     closest_df = proximity._get_closest_arcpy_multithreaded(block_group_points_df, brand_df, 6,
#                                                             data.usa_network_dataset)
#     assert closest_df.equals(closest_df_test)


def test_reformat_closest_result_dataframe(closest_df_test):
    reformat_df = proximity.reformat_closest_result_dataframe(closest_df_test)

    out_cols = ['origin_id', 'destination_rank', 'destination_id', 'proximity_traveltime', 'proximity_kilometers',
                'proximity_side_street_left', 'proximity_side_street_right', 'SHAPE']
    assert list(reformat_df.columns) == out_cols


def test_explode_closest_rank_dataframe(closest_df_test):
    reformat_df = proximity.reformat_closest_result_dataframe(closest_df_test)
    expl_df = proximity.explode_closest_rank_dataframe(reformat_df)
    assert isinstance(expl_df, pd.DataFrame)


def test_get_closest_solution():
    closest_df = proximity.get_closest_solution(origins=block_groups, origin_id_fld=block_group_id_field,
                                                destinations=brand_locs, dest_id_fld=brand_id_field)
    cols = ['origin_id', 'destination_rank', 'destination_id', 'proximity_kilometers', 'proximity_minutes',
            'proximity_side_street_left', 'proximity_side_street_right', 'SHAPE']
    assert list(closest_df.columns) == cols


def test_get_closest_solution_from_layers():
    bg_lyr = arcpy.management.MakeFeatureLayer(str(block_groups))[0]
    brand_lyr = arcpy.management.MakeFeatureLayer(str(brand_locs))[0]
    closest_df = proximity.get_closest_solution(origins=bg_lyr, origin_id_fld=block_group_id_field,
                                                destinations=brand_lyr, dest_id_fld=brand_id_field)
    cols = ['origin_id', 'destination_rank', 'destination_id', 'proximity_kilometers', 'proximity_minutes',
            'proximity_side_street_left', 'proximity_side_street_right', 'SHAPE']
    assert list(closest_df.columns) == cols