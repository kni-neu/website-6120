# CS6120 Natural Language Processing Website Repository

This is the repository for the Natural Language Processing course, taught Spring 2025. 

### Repository structure

This repository is created from a Jekyll template, originally forked from [alshedivat/al-folio](https://github.com/alshedivat/al-folio). 

* **`_data/`**. This folder has syllabus information and lab information. They're stored in `*.yml` files.
* **`_projects/`**. Somewhat of a misnomer. It's staff information. This folder houses profiles about each person, including their office hours, a picture, and contact information. At the beginning of each semester, we will need to update this with relevant teaching staff.


### Building this website locally

This repository builds into [`_website`](./_website), translating Markdown into HTML. It will automatically build (if you set up the Git hooks) when you push to Git. To view your website locally, then follow these steps.

1. Fork [kni-neu/website-6120](`https://github.com/kni-neu/website-6120`) and clone into local folder
2. Run `./start-docker.sh`
  * This starts the docker environment, and calls it `neu-cs6120`.
  * To detach without killing your container, type `Ctrl+P` and then `Ctrl-Q`
  * To access this container, `docker exec -it neu-cs6120 /bin/bash`
3. Inside the container, go to the repo folder (e.g., `cd /home/`
4. Install the packages with `bundle install`
5. Make any changes to markdown files (see the repository structure above).
6. Start the server `./start-server.sh`.
  * Any additional changes should automatically update the server.
7. On a browser, go to http://localhost:8080.
  * The port 8080 is whatever you chose to expose


### Creating a website externally on CCS

At the beginning of each semester, we will need to create a public-facing website. 

1. E-mail khoury-systems@northeastern.edu for a website
  - They'll create `/net/course/cs6120s25 (or whatever URL is agreed upon)
  - This folder will also have `/net/course/cs6120-spXX/.www`
1. Clone into local folder and create Docker environment with ./start-docker.sh
  - Edit `baseurl: /cs6120s25` in `_config.yml`
  - Run `./start-server.sh`
  - Commit to repository and let auto-hooks run
1. Login to: `login.khoury.northeastern.edu`
  - Goto `/net/course/cs6120-spXX/.www`, which will appear as http://course.ccs.neu.edu/cs6120s25.
  - Clone https://github.com/kni-neu/website-6120.git
  - Branch and checkout `gh-pages`
