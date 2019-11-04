# BA-Tools

BA-Tools, shorthand for ArcGIS Business Analyst Tools is a collection of resources combined into a succinct Python package streamlining the process of performing analysis using quantitative geographic and machine learning methods.

## Quickstart

For the impatient, you can install this in your local environment using conda. However, it is highly recommended to use this in conjunction with [GeoAI-Retail](https://github.com/knu2xs/geoai-retail) to dramatically streamline your entire analysis workflow.

```conda install -c knu2xs ba-tools```

## Background

BA-Tool dramatically streamlines the process of data munging to be able to build a model using Machine Learning. This is a supporting package for [GeoAI-Retail](https://github.com/knu2xs/geoai-retail). 

BA-Tools facilitates quantitatively considering the complex interaction of geographic factors using machine learning to perform analyses for deriving insights into human behavior - most notably, human behavior as it relates to retail. Especially when used in conjunction with GeoAI-Retail, BA-Tools dramatically streamlines the process of performing the requisite feature engineering using sound Geographic methods for retail forecasting empirically taking into account the effects of Geographic relationships. 

It is important to note, this package offers no opinion or guidance on creation of the model. Rather, it facilitates the process of data creation, tapping into the vast data and Geographic analysis capabilities of ArcGIS to automate much of  the feature engineering required prior to model training. Further, with this model created, this module enables inferencing using the created model to evaluate the effects of adding or removing a location from the store network.

## Current State

Currently, only analysis using __ArcGIS Pro__ with the __Business Analyst__ extension using __locally installed United States data__ is supported. If you dig into the package, you will find some functions supporting using REST services, but I have yet to get this workflow working reliably. Consequently, for now, it is dependent on ArcPy and locally installed ArcGIS Business Analyst data for the United States. Depending on what use cases we run across, and have to support, international data and even full REST based analysis (not requiring ArcPy) may be supported in the future. Currently though, it is not.

# License - [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

Copyright 2019 Esri

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
