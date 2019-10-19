# BA-Tools

BA-Tools, shorthand for ArcGIS Business Analyst Tools is a collection of resources combined into a succinct Python package streamlining the process of performing analysis using quantitative geographic and machine learning methods.

## Quickstart

For the impatient, you can install this in your local environment using pip.

```pip install ba-tools```

## Background

BA-Tool dramatically streamlines the process of data munging to be able to build a model using Machine Learning. Once created, the model can be used to perform inferencing to evaluate hypothetical scenarios of adding or removing locations. 

It is important to note, this package offers no opinion or guidance on creation of the model. Rather, it facilitates the process of data creation, tapping into the vast data and Geographic analysis capabilities of ArcGIS to automate much of  the feature engineering required prior to model training. Further, with this model created, this module enables inferencing using the created model to evaluate the effects of adding or removing a location from the store network.

## Analysis Workflow

BA-Tools is a set of tools I initially created to facilitate retail analysis. BA-Tools facilitates quantitative consideration  using machine learning of the complex interaction between geographic factors when attempting to perform analyses to derive insights into human behavior - most notably, human behavior as it relates to retail.

This type of analysis typically is performed using the idea of a trade area based on the assumption people within this trade area distance are those who are most likely will visit and patronize this location. This paradigm however, only takes into consideration one geographic factor, the distance or drive time  potential customers are away from the store. 

This paradigm, when we attempted to tackle the more complex geographic interactions, proved inadequate and limiting. What happens when stores of the same brand are closer than this? What happens with a high density of competition is in the area? How does population density affect the trade area? What happens in high population density areas where driving no longer is the primary mode of transportation? What about areas of incredibly low population density, rural areas, where longer travel distances are considered normal? What about mixed population density areas where the trade area could encompass suburban, exurban and rural, where acceptable travel distances vary? What about the areas where the demographic is right, but the distance is longer, where the customers are willing to travel, but are excluded since falling outside of the defined trade area distance?

These are just a few of the confounding questions leading to a completely new method of for location analysis not by starting at the retail location, but rather starting at the customer location, and considering _all_ the factors (at least all we can think of) potentially affecting the customers' retail decision making process.

Modeling from the customers' perspective, from every single customer's individual perspective obviously does not scale. However, we can model at the block group level of geography since, at this level of geography, these areas are demographically homogeneous - everybody is the same. Therefore, we perform analysis from the perspective of these areas to model the perspective of the customer. This process starts with data preparation.

First, we simply need to know as much as possible about the people living in these origin areas, typically US Census Block Groups. With Esri's Business Analyst, we have access to just over 7,800 demographic variables. Hence, the first step is to create a table of these 7,800 variables we can relate back to these origin areas through a unique identifier.

Second we calculate the proximity from these customer origin areas to the closest retail locations for your brand. This is plural, your _locations_, we take into account not just the closest, but the second, third and fourth - up to the maximum number you decide is relevant. Next, we do the same thing for competitor locations, again up to the maximum number deemed relevant.

Third, we assemble the demographics and the proximity metrics into a single very large table with a single row for each origin area. This is the raw table we then can use to train a model.

## Current State

Currently, only analysis using __ArcGIS Pro__ with the __Business Analyst__ extension using __locally installed United States data__ is supported. If you dig into the package, you will find some functions supporting using REST services, but I have yet to get this workflow working reliably. Consequently, for now, it is dependent on ArcPy and locally installed ArcGIS Business Analyst data for the United States. Depending on what use cases we run across, and have to support, international data and even full REST based analysis (not requiring ArcPy) may be supported in the future. Currently though, it is not.

# License - [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

Copyright 2019 Esri

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
