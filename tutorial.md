# Tutorial

This tutorial provides a wlakthrough of the unique capabilities of autoTS. autoTS is based on the [WINGS](https://www.wings-workflows.org) workflow system, which uses semantic constraints to reason about worklfows and as a result it can assist a user to create valid workflow.

To learn the basics about running workflows in WINGS, follow [this tutorial](https://www.wings-workflows.org/tutorial/tutorial.html).

## Table of content
* [Running a workflow](#run)
* [Understanding your results](#results)
* [Uploading data](#data)

## <a name='run'> Running a workflow in autoTS</a>

autoTS comes with several abstract templates for the analysis of time series data. These templates represent strategies used by researchers to analyze their data that can be instantiated with various methods. An example of a workflow strategy for spectral analysis is presented below:

<img src="images/AutoTS.jpg" alt="autoTs_proofofconcept" width="400" />

The methods are available through the [Pyleoclim](https://pyleoclim-util.readthedocs.io/en/stable/) Python package.

* To access a workflow templates, select `Analysis` -> `Run Workflows` in the WINGS portal:

<img src="images/wing_home.png" alt="WINGS-access templates" width="400" />

* Double-click on an abstract template:

<img src="images/Wings-Template.png" alt="Wings Template page" width="400" />

* The template represents the strategy for spectral analysis. Each of the grey box represent a type of function that can be performed (i.e., detrend). The blue boxes represent data flow in between the functions and the green boxes, the parameters.

* You can run workflows by specifying an input timeseries, then click on `Plan Workflow`. The following window will appear:

<img src="images/plan_workflow.png" alt="Wings Template page" width="400" />

* Under templates, you can select the executable workflow that you wish to run, which has been instanciated with proper methods. WINGS can reason over constraints. In our example, one such constraints is that the Lomb-Scargle method doesn't require evenly-spaced data, therefore the LinearInterpolation is greyed out. You can run the workflow by clicking `Run Selected Workflow`.
