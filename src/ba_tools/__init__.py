"""
Business analyst provides a single interface for interacting
with ArcGIS Business Analyst - whether accessing Business Analyst
functionality locally or remotely. Local access is through
ArcGIS Pro with the Business Analyst extension and locally installed
data. Remote access is through a connection to a Web GIS through a GIS
object instance providing access to either ArcGIS Enterprise with
Business Analyst or ArcGIS Online.

!!! note
    Accessing enrich using ArcGIS Online _does_ consume credits.

"""
__version__ = "0.9.1.dev0"

from .__main__ import BusinessAnalyst, Country

__all__ = ["BusinessAnalyst", "Country"]
