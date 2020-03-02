import os
from pathlib import Path
import sys

from arcgis import GeoAccessor
from arcgis.geometry import Geometry
import numpy as np
import pandas as pd
import pytest

# facilitate local resource imports
project_dir_str = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
src_dir_str = project_dir_str/'src'
sys.path.insert(0, str(src_dir_str))
from ba_tools import data, proximity, utils

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'

brand_locs = gdb/'locations_brand'
brand_id_field = 'LOCNUM'

comp_locs = gdb/'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir/'origin_demographics.csv'
nearest_brand_csv = data_dir/'nearest_locations.csv'

scratch_dir = data_dir/'scratch'

closest_parquet = data_dir/'closest_test.parquet'

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
    assert closest_df.equals(closest_df_test)


def test_get_closest_df_arcpy_nex(block_group_points_df, brand_df, closest_df_test):
    closest_df = proximity._get_closest_df_arcpy_nex(block_group_points_df, brand_df, 6, data.usa_network_dataset)
    assert closest_df.equals(closest_df_test)


def test_reformat_closest_result_dataframe(closest_df_test):
    reformat_df = proximity.reformat_closest_result_dataframe(closest_df_test)

    out_cols = ['origin_id', 'destination_rank', 'destination_id', 'proximity_traveltime', 'proximity_kilometers',
                'proximity_side_street_left', 'proximity_side_street_right', 'SHAPE']
    assert list(reformat_df.columns) == out_cols


def test_explode_closest_rank_dataframe(closest_df_test):
    reformat_df = proximity.reformat_closest_result_dataframe(closest_df_test)
    expl_df = proximity.explode_closest_rank_dataframe(reformat_df)
    assert isinstance(expl_df, pd.DataFrame)
