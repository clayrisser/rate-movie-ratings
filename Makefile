CWD := $(shell pwd)

.PHONY: all
all: clean deps dist

.PHONY: start
start: env
	@env/bin/python server.py

env:
	@virtualenv env
	@env/bin/pip install -r ./requirements.txt
	@echo ::: ENV :::

dist: env
	@python setup.py sdist
	@python setup.py bdist_wheel
	@echo ::: DIST :::

.PHONY: publish
publish:
	@twine upload dist/*
	@echo ::: PUBLISHED :::

.PHONY: freeze
freeze:
	@env/bin/pip freeze > ./requirements.txt
	@echo ::: FREEZE :::

.PHONY: clean
clean: clean_data
	-@ rm -rf ./*.pyc ./*/*.pyc ./*/*/*.pyc ./env/ ./*.log ./*.log.* ./nails.egg-info/ ./dist/ ./build/ &>/dev/null || true
	@echo ::: CLEAN :::
