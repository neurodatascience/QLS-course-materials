QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 16, 2024

# Exercise 4: Put both your figures in subplots in the same figure

## Goal
In the last two exercises, you created a plot of summary statistics and a corresponding plot with all the participant data available.
In this exercise, you will put both of those figures in the same subplot.

## Data
- [`../../data/ABIDE_paper_table_1.png`](../../data/ABIDE_paper_table_1.png) contains an image of the table with the summary statistics
- [`../../data/participants_nbsub-200.tsv`](../../data/participants_nbsub-200.tsv) contains the participant-level data from the ABIDE dataset
- [`../../data/ABIDE_LEGEND_V1.02.pdf`](../../data/ABIDE_LEGEND_V1.02.pdf) contains a table with the meanings of the column names in the .tsv

## Task
1. Combine your code from the past two tasks so they're in the same figure, in separate subplots.

## Optional: Use AI to navigate subplot layout

Subplot syntax in matplotlib has several options and is easy to get wrong. This is a good place to use AI as a documentation shortcut. Try asking:

> "In matplotlib, what is the difference between `plt.subplots()`, `gridspec`, and `subplot_mosaic`? Which would you recommend for combining two side-by-side figures?"

Then ask it to generate the layout code for your specific case and adapt it to fit your figures from exercises 2 and 3.

Once you have a combined figure, you can also ask AI to critique it:
> "Here are two subplots side by side: one shows summary statistics, the other shows participant-level data for the same variables. What should I pay attention to so the two panels are visually comparable?"
