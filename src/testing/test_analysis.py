import os
from pathlib import Path
import sys

from arcgis import GeoAccessor
from arcgis.geometry import Geometry
import numpy as np
import pandas as pd
import pytest

# facilitate local resource imports
dir_src = Path(__file__).parent.parent
dir_project = dir_src.parent
sys.path.insert(0, str(dir_src))
from ba_tools import analysis

data_dir = dir_project/'data'/'test'
gdb = data_dir/'test.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'

brand_locs = gdb/'locations_brand'
brand_id_field = 'LOCNUM'

comp_locs = gdb/'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir/'origin_demographics.csv'
nearest_brand_csv = data_dir/'nearest_locations.csv'

scratch_dir = data_dir/'scratch'

if not scratch_dir.exists():
    scratch_dir.mkdir()

@pytest.fixture
def near_df():
    nearest_df = pd.read_csv(str(data_dir/'nearest_locations.csv'))
    return nearest_df

@pytest.fixture
def new_dest():
    geom = Geometry({'x': -122.7342423, 'y': 45.4383307, 'spatialReference': {'latestWkid': 4326, 'wkid': 4326}})
    return geom


def test_get_add_new_closest_dataframe(near_df, new_dest):
    updated_nearest_df = analysis.get_add_new_closest_dataframe(
        origins=block_groups,
        origin_id_field=block_group_id_field,
        destinations=brand_locs,
        destination_id_field=brand_id_field,
        closest_table=near_df,
        new_destination=new_dest
    )
    # 1 is the new id assigned to the added destination; hence, a decent test
    assert('1' not in near_df.destination_id_01.values and '1' in updated_nearest_df.destination_id_01.values)


def test_create_origin_destination_customer_dataframe_parquet():

    dir_raw = data_dir.parent/'raw'
    trips_file = dir_raw/'trips.parquet'
    trips_x = 'coord_x'
    trips_y = 'coord_y'
    trips_dest_id = 'store_id'
    keep_prefix = 'trips_'

    od_df = analysis.create_origin_destination_customer_dataframe(
        customer_points=trips_file,
        customer_destination_id_field=trips_dest_id,
        customer_x_field=trips_x,
        customer_y_field=trips_y,
        customer_keep_field_prefix=keep_prefix
    )

    assert isinstance(od_df, pd.DataFrame)


def test_create_origin_destination_customer_dataframe_csv():

    dir_raw = data_dir.parent/'raw'
    trips_file = dir_raw/'trips.csv'
    trips_x = 'coord_x'
    trips_y = 'coord_y'
    trips_dest_id = 'store_id'
    keep_prefix = 'trips_'

    od_df = analysis.create_origin_destination_customer_dataframe(
        customer_points=trips_file,
        customer_destination_id_field=trips_dest_id,
        customer_x_field=trips_x,
        customer_y_field=trips_y,
        customer_keep_field_prefix=keep_prefix
    )

    assert isinstance(od_df, pd.DataFrame)