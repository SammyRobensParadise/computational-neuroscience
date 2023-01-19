

# makefile


.PHONY: help


help:
	@echo "--- Options ---"
	@echo "markdown ...... generates markdown files from  *.ipynb files"

MYDIR = .

markdown:
	jupyter nbconvert ./tutorials/regression_tutorial.ipynb --to markdown --output regression-tutorial.md
	jupyter nbconvert ./tutorials/mlp_tutorial.ipynb --to markdown --output mlp-tutorial.md