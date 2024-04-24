# DATA

We use the ABIDE dataset as given by Nilearn.

For more information about this dataset's structure:

- http://preprocessed-connectomes-project.org/abide/
- http://www.childmind.org/en/healthy-brain-network/abide/

## Content

- `build_datasets.py` is the python script that was used to generate the data files.
  You can edit `n_subjects` in this script to generate datasets with more participants.

- `participants.tsv`: description of all the participants in the ABIDE dataset.
  To know what each columns represent, check:
  - This
    [PDF](http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf)
    describing the phentypic columns.
  - [This page](http://preprocessed-connectomes-project.org/abide/quality_assessment.html)
    for more details on the columns regarding quality control.

- `"abide_nbsub-{n_subjects}_atlas-{parcellation}_meas-correlation_relmat.tsv"`
  are datasets containing the connectitivity between brain regions
  of the participants of the ABIDE dataset.

- `utils.py` is module with functions to help access some of the data.

Can add the `2024` folder to the python path to use it.

```python
import sys
sys.path.insert(0, "/home/remi/github/origami/QLS-course-materials/Lectures/2024/")

from data.utils import data_loader

data, participants = data_loader()
```
