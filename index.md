## Table of content
* [What is autoTS?](#what)
* [Why autoTS?](#why)
* [Resources](#resources)
* [Contact](#contact)
* [Disclaimer](#disclaimer)

## <a name='what'> What is autoTS?</a>

autoTS is an automated machine learning system for time series analysis. autoTS is conceived around three key ideas:
* To capture expert knowledge as semantic workflow templates that represent common multi-step analysis strategies and constraints, 
* To automatically characterize datasets to extract properties that will affect the choice of analytic methods (e.g., the data has missing values) to instantiate those workflow templates, 
* To curate a catalog of primitive pre-processing and analysis functions for time-series data that map to workflow steps and are annotated with preconditions (e.g., evenly-spaced time series is required), performance (e.g., bias in the low-frequency domain), and other constraints that will be used to generate solutions.

## <a name='why'> Why autoTS?</a>

Time series analysis can be useful to investigate how a given variable (economic or otherwise) changes over time, to identify recurring patterns, new trends, and scaling behaviors. It can also be used to correlate two events in time and in forecasting by using information regarding historical values and associated patterns to predict future changes. Numerous methods have been developed in the past that require extensive expertise. These multi-step methods include not just machine learning algorithms but also data validation and cleaning (e.g. removal of missing values, long- term trend and/or natural log transformation), which are often performed manually and are an integral part of data science. 

Although many signal processing methodologies relevant to time series analysis are widely available in popular packages and libraries (eg, in Matlab , Python , and R ), there are important aspects that require sophisticated expertise:
* Identifying methods and set parameters that are appropriate for a given dataset. 
* Preparing data for analysis. 
* Specifying the null hypothesis.

Consequently, time series analysis is often considered more of an art than a science, a craft that takes years to develop and master. Because few people possess the expertise to process time series data, we have limited capability to analyze from the all the time series data that have been collected. Because sophisticated expertise takes practice, few practitioners are well qualified to do time series data analysis properly.

autoTS capture these strategies and methods and systematically searches through the space of solutions for a given dataset. A researcher with limited knowledge about time series analysis is able to use autoTS to efficiently generate valid solutions for their questions. 

## <a name='resources'> Resources </a>

### Software

autoTS relies on (and has extended) the [WINGS workflow system](https://www.wings-workflows.org). 
The methods are available through the [Pyleoclim package](https://github.com/LinkedEarth/Pyleoclim_util/tree/master).
Workflow strategies and test datasets are available on our [GitHub repository](https://github.com/KnowledgeCaptureAndDiscovery/autoTS).

### Publications

Deborah Khider, Pratheek Athreya, Varun Ratnakar, Yolanda Gil, Feng Zhu,Myron Kwan, and Julien Emile-Geay. 2020. Towards Automating Time SeriesAnalysis for Paleogeosciences. InMileTS ’20: 6th KDD Workshop on Mining and Learning from Time Series, August 24th, 2020, San Diego, California, USA.ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/1122445.xxxxx1 [[PDF]](https://github.com/KnowledgeCaptureAndDiscovery/autoTS/blob/master/manuscripts/KDD_TimeSeries_Workshop.pdf)

## <a name='contact'> Contact </a>
For questions about autoTS, please contact [Deborah Khider](mailto:khider@usc.edu) or [Yolanda Gil](mailto:gil@isi.edu)

## <a name='disclaimer'> Disclaimer </a>
This research is funded by JP Morgan Chase & Co. Any views or opinions expressed herein are solely those of the authors listed, and may differ from the views and opinions expressed by JP Morgan Chase & Co. or its affilitates. This material is not a product of the Research Department of J.P. Morgan Securities LLC. This material should not be construed as an individual recommendation of for any particular client and is not intended as a recommendation of particular securities, financial instruments or strategies for a particular client. This material does not constitute a solicitation or offer in any jurisdiction.