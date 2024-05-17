QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 5, 2023

# Exercise 4: Put both your figures in subplots in the same figure

## Goal
In the last two exercises, you created a plot of summary statistics and a corresponding plot with all the participant data available.
In this exercise, you will put both of those figures in the same subplot.

## Data
- [`../../data/ABIDE_paper_table_1.png`](../../data/ABIDE_paper_table_1.png) contains an image of the table with the summary statistics
- [`../../data/participants_nbsub-200.tsv`](../../data/participants_nbsub-200.tsv) contains the participant-level data from the ABIDE dataset
- [`../../data/ABIDE_LEGEND_V1.02.pdf`](../../data/ABIDE_LEGEND_V1.02.pdf) contains a table with the meanings of the column names in the .tsv

## Task
1. Combine your code from the past two tasks to be in the same figure, in separate subplots.

## HINT 1
These are some steps you could follow
1. Save a .py file
2. Import the Path object from pathlib
3. Import the pyplot sublibrary of matplotlib
4. Import the pandas library
5. Use the Path object to define the path to the .csv
6. Use pandas to read the csv
7. Create a blank Figure and Axes
8. Use the Path object to define the path to where you want to save the figure
9. Save the figure
10. Run the code
11. Open the figure so you can see how it changes when you adjust the code
12. Use explicit coding to draw and customize the subplots.
