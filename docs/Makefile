# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Processing .pyi files and backing up related files..."
	@find .. -name "*.pyi" | while read pyi_file; do \
		base_name="$${pyi_file%.pyi}"; \
		pxd_file="$$base_name.pxd"; \
		pyx_file="$$base_name.pyx"; \
		py_file="$$base_name.py"; \
		if [ -f "$$pxd_file" ]; then \
			mv "$$pxd_file" "$$pxd_file.bak"; \
			echo "Backed up $$pxd_file to $$pxd_file.bak"; \
		fi; \
		if [ -f "$$pyx_file" ]; then \
			mv "$$pyx_file" "$$pyx_file.bak"; \
			echo "Backed up $$pyx_file to $$pyx_file.bak"; \
		fi; \
		if [ -f "$$py_file" ]; then \
			mv "$$py_file" "$$py_file.bak"; \
			echo "Backed up $$py_file to $$py_file.bak"; \
		fi; \
		cp "$$pyi_file" "$$py_file"; \
		echo "Copied $$pyi_file to $$py_file"; \
	done
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Restoring original files..."
	@find .. -name "*.pyi" | while read pyi_file; do \
		base_name="$${pyi_file%.pyi}"; \
		py_file="$$base_name.py"; \
		pxd_file="$$base_name.pxd"; \
		pyx_file="$$base_name.pyx"; \
		if [ -f "$$py_file" ]; then \
			rm "$$py_file"; \
			echo "Removed temporary $$py_file"; \
		fi; \
		if [ -f "$$pxd_file.bak" ]; then \
			mv "$$pxd_file.bak" "$$pxd_file"; \
			echo "Restored $$pxd_file"; \
		fi; \
		if [ -f "$$pyx_file.bak" ]; then \
			mv "$$pyx_file.bak" "$$pyx_file"; \
			echo "Restored $$pyx_file"; \
		fi; \
		if [ -f "$$py_file.bak" ]; then \
			mv "$$py_file.bak" "$$py_file"; \
			echo "Restored $$py_file"; \
		fi; \
	done
	@echo "Documentation build process completed."