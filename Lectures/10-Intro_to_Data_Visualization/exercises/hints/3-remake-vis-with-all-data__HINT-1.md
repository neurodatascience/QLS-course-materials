QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 5, 2023

# Exercise 3: Remake your visualization with all the data
(not just the summary statistics)

## Goal
In the last two exercises, you planned and created a figure using summary statistics from a table.
In this exercise, you will re-design and encode your figure from exercise 1, with all the participant-level data. 

## Data
- `Phenotypic_V1_0b_preprocessed1.csv` contains the full participant-level data from the ABIDE dataset
- `ABIDE_LEGEND_V1.02.pdf` contains a table with the meanings of the column names in the .csv

## Task
1. Plan whether/how to change your figure 
2. Write a python script to create the figure

## HINT 1 
These are some steps you could follow
1. Save a .py file
2. Import the pyplot sublibrary of matplotlib
3. Import the pandas library
4. Use pandas to read the csv data into a DataFrame
5. Print the dataframe and/or look at it in a spreadsheet program (e.g., Excel or LibreOffice Calc) in order to see which column(s) you need.
6. Create a blank Figure with Axes
7. Save the figure
8. Run the code
9. Open the figure so you can see how it changes when you adjust the code
10. Use explicit coding to draw a figure with the data in the DataFrame.