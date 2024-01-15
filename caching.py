"""
This module sets up a Joblib cache to be accessed by other modules
in order to cache on disk the results of expensive computations
by simply annotating functions with `@MEM.cache`
"""

import joblib
import os

_ENV_VAR = "WORKGEN_CACHE_DIR"
if _ENV_VAR not in os.environ:
    import logging

    CACHE_DIR = ".joblib-cache"
    logging.warn("%s is not set, using %s for the joblib cache" % (_ENV_VAR, CACHE_DIR))
else:
    CACHE_DIR = os.environ[_ENV_VAR]

MEM = joblib.Memory(".joblib-cache")
