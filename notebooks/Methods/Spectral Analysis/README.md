Spectral Analysis

The following notebooks were used for the development of the spectral method incorporated into Pyleoclim:

Completed work:

* analytical_spec: computes the analytical spectral for specific timeseries and compare to the method available in Pyleoclim
* astropy: Looks at the various algorithm available in the [astropy](https://www.astropy.org) package. Very basic testing didn't show a difference for our application and we proceeded with Scipy for Lomb-Scargle.
* Lomb Scargle Performance on Evenly Spaced Signal.ipynb: Looks at the implementation of Lomb-Scargle in Pyleoclim with respect to WOSA implementation and use.
* lomb_scargle_vs_wwz_analytical_benchmarks.ipynb: Evaluates the performance of the Lomb-Scargle periodogram with respect to analytical solution on specific signals.
* Lomb-Scargle Performance_RvsPython.ipynb: Benchmarks the Lomb-Scargle algorithm as implemtented in GeochronR vs Pyleoclim
*

In Progress

* gap_spectral: Notebook to look at the effect of gaps on the analysis.
* run_algs.py (note: not a Notebook): runs various algorithms and test accuracy for the peaks
