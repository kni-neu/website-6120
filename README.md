# CS6120 Natural Language Processing Website Repository

This is the repository for the Natural Language Processing course, taught Spring 2025. In order to use this repository, you will need to do the following:

1. Obtain a website address from Khoury Systems. Do so by e-mailing khoury-systems@northeastern.edu, with the request. They will create a folder for us in `/net/course/`, which will likely be of the format `cs6120-spXX`. 
1. Clone the repository into a local folder. In order to test it, you can use the specified Dockerfile by running `start-docker.sh`, pulling the appropriate folders.
1. Change `_config.yml` to have the appropriate path in the baseurl. In this case, you would just type `baseurl: /cs6120-spXX`
1. Compile with `./start-server.sh` to see if it runs. Otherwise, when you commit, it will go to this repository and kick off jobs in the branch `gh-pages`.
1. Go to your web folder on `login.khoury.northeastern.edu`. This will likely be `/net/course/cs6120-spXX/.www`, where the `.www` folder will be what appears on http://course.ccs.neu.edu/cs6120-spXX.
1. Clone the repository into this folder. (You may copy everything to .www once cloned, but make sure you have the .git and .gitignore files.)
1. Checkout the `gh-pages` branch. This will be the web-ready version. Keep pulling from here. 
