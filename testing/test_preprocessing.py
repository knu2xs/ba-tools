import os
from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ba_tools import preprocessing

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'
brand_locs = gdb/'locations_brand'
brand_id_field = 'LOCNUM'
comp_locs = gdb/'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir/'origin_demographics.csv'
scratch_dir = data_dir/'scratch'

if not scratch_dir.exists():
    scratch_dir.mkdir()


def test_origins_to_dataframe():
    origins_to_dataframe = preprocessing.OriginGeographyFeatureClassToDataframe(
        geography_id_field=block_group_id_field
    )
    origin_df = origins_to_dataframe.fit_transform(block_groups)
    assert isinstance(origin_df, pd.DataFrame)


def test_add_demographics():
    add_demographics_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddDemographicsToOriginDataframe(
            origin_geography_layer=block_groups,
            geography_id_field=block_group_id_field,
            interim_data_directory=scratch_dir
        ))
    ])
    df = add_demographics_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'origin_demographics.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())


def test_add_nearest_locations():
    add_nearest_loc_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddNearestLocationsToOriginDataframe(
            origin_geography_layer=block_groups,
            origin_id_field=block_group_id_field,
            location_layer=brand_locs,
            location_id_field=brand_id_field,
            destination_count=6,
            interim_data_directory=scratch_dir,
            clobbber_previous_results=False
        ))
    ])
    df = add_nearest_loc_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'nearest_locations.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())


def test_add_nearest_competition_locations():
    add_nearest_comp_loc_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddNearestCompetitionLocationsToOriginDataframe(
            origin_geography_layer=block_groups,
            origin_id_field=block_group_id_field,
            competition_location_layer=comp_locs,
            competition_location_id_field=comp_id_field,
            destination_count=6,
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=False
        ))
    ])
    df = add_nearest_comp_loc_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'nearest_locations.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())
