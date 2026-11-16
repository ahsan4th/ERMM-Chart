"""ERMM-Chart: EWMA robust Max-M control chart for the joint monitoring of the
mean vector and the covariance matrix of multivariate individual observations.

Reference implementation accompanying the manuscript by Radjid, Ahsan,
Wibawati and Lee. The Phase I estimator reproduces robustbase::covMcd
(alpha = 0.75, reweighting at the 0.975 chi-square quantile, consistency and
small-sample corrections).
"""
from .charts import MU_M, SIGMA_M, ermm_path, max_m_components, mewma_path, ucl_from_L, L_from_ucl
from .estimators import PhaseIEstimate, classical_estimate, estimate, h_alpha_n, reweighted_fast_mcd

__version__ = "1.0.0"
__all__ = ["MU_M", "SIGMA_M", "ermm_path", "max_m_components", "mewma_path", "ucl_from_L",
           "L_from_ucl", "PhaseIEstimate", "classical_estimate", "estimate", "h_alpha_n",
           "reweighted_fast_mcd", "__version__"]
