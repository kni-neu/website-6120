#!/bin/bash

docker run -p 8080:8080 -v $PWD/../:/home -it amirpourmand/al-folio /bin/bash
