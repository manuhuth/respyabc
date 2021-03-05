"""General configuration for respyabc."""
from pathlib import Path

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parent
