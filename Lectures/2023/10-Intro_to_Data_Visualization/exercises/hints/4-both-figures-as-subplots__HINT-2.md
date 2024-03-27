QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 5, 2023

# Exercise 4: Put both your figures in subplots in the same figure

## Goal
In the last two exercises, you created a plot of summary statistics and a corresponding plot with all the participant data available.
In this exercise, you will put both of those figures in the same subplot.

## Data
- `ABIDE_paper_table_1.png` contains an image of the table with the summary statistics
- `Phenotypic_V1_0b_preprocessed1.csv` contains the full participant-level data from the ABIDE dataset
- `ABIDE_LEGEND_V1.02.pdf` contains a table with the meanings of the column names in the .csv

## Task
1. Combine your code from the past two tasks to be in the same figure, in separate subplots.

## HINT 2
These are some steps you could follow
1. Save a .py file
2. Import the Path object from pathlib
   - `from pathlib import Path`
3. Import the pyplot sublibrary of matplotlib
   - `import matplotlib.pyplot as plt`
4. Import the pandas library
   - `import pandas as pd`
5. Import the seaborn library
6. Use the Path object to define the path to the .csv
   - `Path(__file__).resolve().parent / "data" / "Phenotypic_V1_0b_preprocessed1.csv"`
7. Use pandas to read the csv
   - `df = pd.read_csv(<PATH>)`
8. Some values in the csv ars -999. You'll want to delete these in the column that you need.
   - `df = df[["DX_GROUP", variable]]`
   - `df = df.dropna()`
   - `df = df[df[variable] > 0]`
9. Create a blank Figure and Axes
   - `fig, axs = plt.subplots(1, 2,)`
10.  Use the Path object to define the path to where you want to save the figure
   - `Path(__file__).resolve().parent / 4-both-figures-as-subplots.png`
11.  Save the figure
12.  Run the code
13.  Open the figure so you can see how it changes when you adjust the code
14.  Use explicit coding to draw and customize the subplots.
    - `axs[0].bar(x, height)`
    - `sns.violinplot(df, x, y, ax=axs[1])`