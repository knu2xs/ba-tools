# BA-Tools

BA-Tools, shorthand for ArcGIS Business Analyst Tools is a collection of resources combined into a succinct Python package streamlining the process of performing analysis combining quantitative geographic and machine learning methods.

## Quickstart

For the impatient, you can install this in your local environment using conda. It is a publicly available as an [installable Conda package](https://anaconda.org/knu2xs/ba-tools). However, it is highly recommended to use this in conjunction with [GeoAI-Retail](https://github.com/knu2xs/geoai-retail) to dramatically streamline the entire analysis workflow.

``` bash
> conda install -c knu2xs ba-tools
```

## Overview and Features

BA-Tool dramatically streamlines the process of data munging to be able to build a model using Machine Learning. This is a supporting package for [GeoAI-Retail](https://github.com/knu2xs/geoai-retail). 

BA-Tools facilitates quantitatively considering the complex interaction of geographic factors using machine learning to perform analyses for deriving human behavior insights - most notably, human behavior as it relates to retail. Especially when used in conjunction with [GeoAI-Retail](https://github.com/knu2xs/geoai-retail), BA-Tools dramatically streamlines the process of performing the requisite feature engineering using sound Geographic methods for retail forecasting. 

It is important to note, this package offers no opinion or guidance on creation of the model. Rather, it facilitates the process of [feature engineering](https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a), tapping into the vast data and Geographic analysis capabilities of ArcGIS to automate much of the feature engineering required to quantitatively create Geographic factors to include in model training. With this model created, BA-Tools module also enables inferencing using the created model to evaluate the effects of adding or removing a location from the retail landscape.

## Requirements

Currently, only analysis using [__ArcGIS Pro__](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) with the [__Business Analyst__](https://www.esri.com/en-us/arcgis/products/arcgis-business-analyst/applications/desktop) extension using __locally installed United States data__ is supported. Consequently, for now, it is dependent on ArcPy and locally installed ArcGIS Business Analyst data for the United States. Depending on what use cases we run across, and have to support, international data and even full REST based analysis (not requiring ArcPy) may be supported in the future. Currently though, it is not.

## Issues

Find a bug or want to request a new feature?  Please let us know by submitting an issue.

## Contributing

Esri welcomes contributions from anyone and everyone. Please see our [guidelines for contributing](https://github.com/esri/contributing).

# License - [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

Copyright 2020 Esri

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
