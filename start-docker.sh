#!/bin/bash

# Testing
docker run -p 8060:8080 -v $PWD:/home --rm -it --name neu-cs6120 --platform linux/amd64 amirpourmand/al-folio /bin/bash
# docker run -p 8080:8080 -v $PWD:/home -it --name neu-cs6120 --platform linux/amd64 amirpourmand/al-folio /bin/bash

