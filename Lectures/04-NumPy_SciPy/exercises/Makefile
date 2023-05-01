solutions := $(filter-out $(wildcard solutions/_*.py),$(wildcard solutions/*.py))
questions := $(patsubst solutions/%,questions/%, $(solutions))

.PHONY: all clean

all: $(questions)

questions/%.py: solutions/%.py
	sed '/^ *# \?TODO_BEGIN/,/^ *# \?TODO_END/d' $< > $@

clean:
	rm -f $(questions)
