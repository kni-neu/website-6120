# CS6120 Natural Language Processing Website Repository

This is the repository for the Natural Language Processing course, taught Spring 2025. In order to use this repository, you will need to do the following:

1. E-mail khoury-systems@northeastern.edu for a website
  a. They'll create `/net/course/cs6120-spXX`
  a. This folder will also have `/net/course/cs6120-spXX/.www`
1. Clone into local folder and create Docker environment with ./start-docker.sh
  a. Edit `baseurl: /cs6120-spXX` in `_config.yml`
  a. Run `./start-server.sh`
  a. Commit to repository and let auto-hooks run
1. Login to: `login.khoury.northeastern.edu`
  a. Goto `/net/course/cs6120-spXX/.www`, which will appear as http://course.ccs.neu.edu/cs6120-spXX.
  a. Clone https://github.com/kni-neu/website-6120.git
  a. Branch and checkout `gh-pages`
