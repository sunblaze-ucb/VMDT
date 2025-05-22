import sys
from pathlib import Path

# workaround for opensora import
sys.path.append(str(Path(__file__).parent / "repo"))

from .model import OpenSora as OpenSora