QLS612 Neuro Data Sciece | Introduction to Data Visualization | Friday, May 5, 2023

# Exercise 2: Encode a simple visualization

## Task
1. Write a python script to create the figure that you planned in exercise 1.
2. Compare your code with the partner who planned the figure with you.

## HINT 1
These are some steps you could follow, and the corresponding functions
1. Save a .py file
2. Import the pyplot sublibrary of matplotlib
   - `import matplotlib.pyplot as plt`
3. Represent your data with variables
   - e.g.,
   - `categories = ['A', 'B']`
   - `magnitudes = [1, 2]`
   - `colors = ['green','red']`
4. Create a blank Figure with Axes
   - `fig, ax = plt.subplots()`
5. Save the figure
   - `fig.savefig(<PATH>)`
6. Run the code
7. Open the figure so you can see how it changes when you adjust the code
8.  Use matplotlib's explicit coding style to draw your figure.
   - e.g., `ax.hist(x=<VARIABLE>, height=<VARIABLE>, color=<VARIABLE>)`
