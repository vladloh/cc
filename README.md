cc
cc's project


# Docker

### Explanation
Dockerfile in this repository allows to create a docker container with all required software to run all research stuff in mrartemev/research. This container is using nvidia-docker stuff for cuda purposes

### Pulling

Now you can simply pull already existing docker images from Docker Hub with 

```docker pull mrartemev/cc```

If you want to build your own docker image proceed to the next thingy below

### Building
To build an image run following code inside repo directory

``` docker build --rm -t <image_name> . ```

For example:

```docker build --rm -t cc:current .```

### Usage
To run docker image run following code inside repo directory

```docker run --rm -v `pwd`:/home/CC --name <name> --runtime nvidia -it -p <port>:8888 <image_name>```

For example:

```docker run --rm -v `pwd`:/home/CC --name cc --runtime nvidia -it -p 8808:8888 mrartemev/cc```

Running this command will mount docker to your repo directory and execute jupyter notebook command inside your docker.

Open this in your browser to work with repo http://localhost(or yours server-id):8888 (or other chosen <port>). After that, you'll be able to run ipython notebooks in /research

Once you finished just do usual `git add`, `git commit`, `git push` to commit changes to our repo

I recommend to run this docker image in something like tmux or screen just in case

If something goes wrong ping me @mrartemev
