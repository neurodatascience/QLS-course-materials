QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 16, 2024

# Exercise 3: Remake your visualization with all the data
(not just the summary statistics)

## Goal
In the last two exercises, you planned and created a figure using summary statistics from a table.
In this exercise, you will re-design and encode your figure from exercise 1, with all the participant-level data.

## Data
- [`../../data/participants_nbsub-200.tsv`](../../data/participants_nbsub-200.tsv) contains the participant-level data from the ABIDE dataset
- [`../../data/ABIDE_LEGEND_V1.02.pdf`](../../data/ABIDE_LEGEND_V1.02.pdf) contains a table with the meanings of the column names in the .tsv

## Task
1. Plan whether/how to change your figure
2. Write a python script to create the figure

## Optional: Use AI to help with data and design decisions

**Understanding the data:** The legend PDF has a lot of columns. Paste the relevant section into an AI and ask:
> "Which of these columns are most relevant for comparing autism symptom severity between groups?"

This saves lookup time and lets you focus on the visualization decisions.

**Choosing a plot type:** Switching from summary statistics to participant-level data opens up new options (violin plots, strip plots, box plots, etc.). Ask AI to argue for one (This is an example question, your question could be different):
> "I want to show the distribution of [variable] split by autism diagnosis vs. control. Compare violin plots, strip plots, and box plots for this purpose — which would you recommend and why?"

Then critically evaluate the argument: do you agree? Does the AI consider your specific message and audience?

**Writing the code:** As in exercise 2, you can use AI to generate code — but make sure you understand what it produces before moving on.
