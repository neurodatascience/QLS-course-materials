# DATA

We use the ABIDE dataset as given by Nilearn.

For more information about this dataset's structure:

- http://preprocessed-connectomes-project.org/abide/
- http://www.childmind.org/en/healthy-brain-network/abide/

## Content

- `build_datasets.py` is the python script that was used to generate the data
  files.

- `participants.tsv`: description of all the participants in
  the datasets To know what each columns represent, check:
  - This
    [PDF](http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf)
    describing the phentypic columns.
  - [This page](http://preprocessed-connectomes-project.org/abide/quality_assessment.html)
    for more details on the columns regarding quality control.
