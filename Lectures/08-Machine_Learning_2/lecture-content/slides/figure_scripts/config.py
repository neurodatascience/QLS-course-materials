import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import cm

TAB10_COLORS = [cm.tab10(i) for i in range(10)]
TAB20_COLORS = [cm.tab20(i) for i in range(20)]

script_name = Path(sys.argv[0]).stem
FIGURES_DIR = Path(__file__).parents[1] / "figures" / "generated" / script_name
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

mpl.rc("text", usetex=True)
# mpl.rc("font", family="serif")
mpl.rc(
    "text.latex",
    preamble=r"\usepackage{eulervm} \usepackage{amssymb} "
    r"\usepackage{amsmath} \usepackage{bm} \usepackage{DejaVuSans} "
    r"\newcommand{\X}{{\mathbold X}} "
    r"\newcommand{\U}{{\mathbold U}} "
    r"\newcommand{\V}{{\mathbold V}} "
    r"\newcommand{\bS}{{\mathbold S}} "
    r"\newcommand{\y}{{\mathbold y}} "
    r"\newcommand{\x}{{\mathbold x}} "
    r"\newcommand{\W}{{\mathbold W}} "
    r"\newcommand{\bH}{{\mathbold H}} "
    r"\newcommand{\bbeta}{{\mathbold \beta}} ",
)
