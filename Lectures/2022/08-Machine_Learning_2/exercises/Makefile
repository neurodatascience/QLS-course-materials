solutions := $(wildcard *_solutions.py)
questions := $(patsubst %_solutions.py,%_questions.py, $(solutions))
all_scripts = $(solutions) $(questions)
notebooks := $(patsubst %.py,notebooks/%.ipynb,$(all_scripts))

.PHONY: all clean

all: $(questions) $(notebooks)

%_questions.py: %_solutions.py
	sed '/^ *# \?TODO_BEGIN/,/^ *# \?TODO_END/d' $< > $@

notebooks/%.ipynb: %.py
	jupytext $< --output $@

clean:
	rm -f notebooks/*.ipynb
	rm -f *_questions.py
