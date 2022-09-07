#Dockerfile vars
pythonver=3.8-slim

#vars
IMAGENAME=emlo2_s2
REPO= mullermuttu
APP_VER = 1.0
IMAGEFULLNAME=${REPO}/${IMAGENAME}:${APP_VER}

.PHONY: help build push all

help:
	    @echo "Makefile arguments:"
	    @echo ""
	    @echo "alpver - Alpine Version"
	    @echo "kctlver - kubectl version"
	    @echo ""
	    @echo "Makefile commands:"
	    @echo "build"
	    @echo "push"
	    @echo "all"

.DEFAULT_GOAL := all

build:
	    @docker build --pull --build-arg PY_VER=${pythonver} -t ${IMAGEFULLNAME} .

push:
	    @docker push ${IMAGEFULLNAME}

all: build push