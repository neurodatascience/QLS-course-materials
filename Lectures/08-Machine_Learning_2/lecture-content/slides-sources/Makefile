.PHONY: all clean clean-all

all: ../08-Machine_Learning_2.pdf


presentation.d: presentation.org
	@sed -n -e '/^.*\[\[file:\(figures\/\(generated\|graphs\)\/[^]]*\)\]\].*/{s//\1/;H};' \
	-e '$${x;s/\n/ /g;s/\(.\+\)/figs := \1/p}' $< > $@
	@ls figure_scripts/*.py | \
		sed 's/\(figure_scripts\/\)\(.*\)\.py/figures\/generated\/\2\/%: \1\2.py figure_scripts\/config.py\n\tpython3 $$</' >> $@

figures/generated/cv_figure_simple.pdf: figure_scripts/show_cv.py
	python3 $< --simple $@

figures/generated/cv_figure_nested.pdf: figure_scripts/show_cv.py
	python3 $< $@

figures/graphs/%.pdf: figure_scripts/%.dot
	figure_scripts/draw_graph.sh $< > $@

clean:
	rm -f presentation.pdf presentation.d

clean-all: clean
	rm -rf figures/generated

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), clean-all)
-include presentation.d
endif
endif

presentation.pdf: presentation.org $(figs)
	emacs $< --batch -f org-beamer-export-to-pdf

../08-Machine_Learning_2.pdf: presentation.pdf
	pdftk F=figures/generated/cv_figure_nested.pdf P=presentation.pdf cat P1-32 F P34-end output $@
