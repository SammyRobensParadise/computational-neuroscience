

# makefile


.PHONY: help


help:
	@echo "--- Options ---"
	@echo "markdown ...... generates markdown files from  *.ipynb files"

MYDIR = .

markdown:
	jupyter nbconvert ./tutorials/regression_tutorial.ipynb --to markdown --output regression-tutorial.md
	jupyter nbconvert ./tutorials/mlp_tutorial.ipynb --to markdown --output mlp-tutorial.md
	jupyter nbconvert ./tutorials/cnn_tutorial.ipynb --to markdown --output cnn-tutorial.md
	jupyter nbconvert ./tutorials/convolution_tutorial.ipynb --to markdown --output convolution-tutorial.md
		jupyter nbconvert ./tutorials/recurrent_network_tutorial.ipynb --to markdown --output recurrent-network-tutorial.md