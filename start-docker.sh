#!/bin/bash

docker run -p 8080:8080 -v $PWD:/home -it --platform linux/amd64 amirpourmand/al-folio /bin/bash
