.. FRETBursts documentation master file, created by
   sphinx-quickstart on Fri Mar 07 15:30:19 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Single-molecule FRET burst analysis
==========================================


.. raw:: html


    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs">
      <div class="row align-items-center">

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Selecting%20FRET%20populations.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/alex_jointplot_fit.png">
          </div>
        </a>

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Background%20estimation.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/hist_bg_fit.png">
          </div>
        </a>

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Plotting%20timetraces%20with%20bursts.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/timetrace_bursts.png">
          </div>
        </a>

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Burst%20Variance%20Analysis.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/BVA_joint.png">
          </div>
        </a>

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%202CDE%20Method.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/2cde_joint.png">
          </div>
        </a>

        <a href="http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20FRET%20histogram%20fitting.ipynb">
          <div class="col-sm-2 thumbnail">
            <img src="_static/fret_hist_fit.png">
          </div>
        </a>

      </div>
    </div>
    <br>



    <div class="container-fluid">
    <div class="row">
    <div class="col-md-6">
    <br>


`FRETBursts <http://opensmfs.github.io/FRETBursts/>`__ is an open-source
python package for burst analysis of freely-diffusing
`single-molecule FRET <https://en.wikipedia.org/wiki/Single-molecule_FRET>`__
data for single and multi-spot experiments. FRETBursts supports both
single-laser and dual-laser alternated excitation (ALEX and PAX)
as well as ns-ALEX (or PIE).

We provide well-tested implementations of state-of-the-art
algorithms for confocal smFRET analysis.
We focus on computational reproducibility,
by using `Jupyter notebook <http://jupyter.org/>`__ based interfaces.

Please send questions or report issue on `GitHub <https://github.com/OpenSMFS/FRETBursts/issues>`__.

.. raw:: html

   </div>
   <div class="col-md-3">
   <h2>Documentation</h2>

* `Introducing FRETBursts <http://tritemio.github.io/smbits/2016/02/19/fretbursts>`__
* :doc:`Installation <getting_started>`
* :doc:`What's new? <releasenotes>`
* `μs-ALEX Tutorial <http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/FRETBursts%20-%20us-ALEX%20smFRET%20burst%20analysis.ipynb>`__
* `List of Jupyter Notebooks <https://github.com/OpenSMFS/FRETBursts_notebooks#fretbursts-notebooks>`__
* :doc:`Reference manual <reference_manual>`

.. raw:: html

   </div>
   <div class="col-md-3">
   <h2>Features</h2>

* `FRETBursts Paper <http://dx.doi.org/10.1101/039198>`__
* :doc:`Burst Search Algorithm <burstsearch>`
* `BVA <http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Burst%20Variance%20Analysis.ipynb>`__
* `2CDE <http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%202CDE%20Method.ipynb>`__
* `Exporting burst data <http://nbviewer.jupyter.org/github/OpenSMFS/FRETBursts_notebooks/blob/master/notebooks/Example%20-%20Exporting%20Burst%20Data%20Including%20Timestamps.ipynb>`__
* `Report an issue <https://github.com/opensmfs/FRETBursts/issues>`__
