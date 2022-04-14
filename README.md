# mb-cog-maps-paper

Repository for the paper "Construction and Use of Cognitive Maps in Model-Based Control" 

Organization:
```yaml
├── src : package containing the python code used for the papers analyses
│   ├── data : code for processing the data and generating demographics information
│   ├── rl : code for the reinforcement learning model
│   └── utils : code for generating figures and statistics, as well as utility functions for use in the data and rl subpackages
├── data : all data analyzed in the paper
│   ├── raw : 
│   │   ├── experiment_1 : data for the initial pilot sample
│   │   └── experiment_2 : data for the main sample reported in the paper
│   └── processed : all processed data
│   │   ├── experiment_1 : data for the initial pilot sample
│   │   └── experiment_2 : data for the main sample reported in the paper
└── paper : all files to generate paper
    └── figs : pdf copies of data figures used in paper
        ├── pilot : figures generated for the supplement
        └── primary : figures in the main paper  
```

## One time setup
Here we describe two options for recreating our computational environment Docker and Conda. Instruction for the docker installation has been copied from [MIND](https://github.com/Summer-MIND/mind-tools) repo
### Docker
1. Install Docker on your computer using the appropriate guide below:
    - [OSX](https://docs.docker.com/docker-for-mac/install/#download-docker-for-mac)
    - [Windows](https://docs.docker.com/docker-for-windows/install/)
    - [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
    - [Debian](https://docs.docker.com/engine/installation/linux/docker-ce/debian/)
2. Launch Docker and adjust the preferences to allocate sufficient resources (e.g. >= 4GB RAM)
3. To build the Docker image, open a terminal window, navigate to your local copy of the repo, and run `docker build . -t mb-cog-maps` (note the first time build might take a while ~15 mins)
4. Use the image to run a container with the repo mounted as a volume so the code and data are accessible.
    - The command below will create a new container that maps the repository on your computer to the `/mnt` directory within the container, so that location is shared between your host OS and the container. Be sure to replace `LOCAL/REPO/PATH` with the path to the cloned repository on your own computer (you can get this by navigating to the repository in the terminal and typing `pwd`).  The below command will also share port `9999` with your host computer, so any Jupyter notebooks launched from *within* the container will be accessible at `localhost:9999` in your web browser
    - `docker run -it -p 9999:9999 --name mb-cog-maps -v /LOCAL/REPO/PATH:/mnt mb-cog-maps`
    - You should now see the `root@` prefix in your terminal. If you do, then you've successfully created a container and are running a shell from *inside*!
5. To launch any of the notebooks, simply enter `jupyter notebook` and copy/paste the link generated into your browser.

#### Using the container after setup
1. Type the following into the terminal: `docker start mb-cog-maps && docker attach mb-cog-maps`
2. Then launch a given notebook from the following list under scripts/notebooks: (01-data-cleaning.ipynb, 02-model-fitting.ipynb, 03-paper-analyses.ipynb, 04-paper-figures.ipynb)
3. There are also supplemental notebooks listed with S## indicating their ordering. 

### Conda
1. Install the anaconda python distribution on your computer using appropriate guide below (I would recommend the command line utility):
    - [OSX](https://docs.anaconda.com/anaconda/install/mac-os/)
    - [Windows](https://docs.anaconda.com/anaconda/install/windows/)
    - [Linux](https://docs.anaconda.com/anaconda/install/linux/)
2. Once anaconda is installed run `conda init` in the terminal
3. Navigate to the repository on your computer and run `conda create -n mb-cog-maps --file requirements.txt`
4. Once the environment is created run `conda activate mb-cog-maps` and then `jupyter notebook`
5. Launch any given notebook under scripts/notebooks folder: (01-data-cleaning.ipynb, 02-model-fitting.ipynb, 03-primary-analyses.ipynb, 04-paper-figures.ipynb)

Please feel free to send me an email at a.b.karagoz@wustl.edu or post an issue if you are having difficulties running any component of this code.