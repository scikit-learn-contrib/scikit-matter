__version__ = "0.1.3"

import logging

LOGGER = logging.getLogger(__name__)

LOGGER.warn(
    "DeprecationWarning: This package has been renamed to scikit-matter (skmatter https://github.com/lab-cosmo/scikit-matter). This package will no longer be maintained and updated. Please install the new package using `pip install skmatter`"
)
