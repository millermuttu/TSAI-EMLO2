[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8324427&assignment_repo_type=AssignmentRepo)
# EMLO V2 - Session 02

## The list of action items to complete the assignment.

- Implementation of pytorch-lightning code for CIFAR10 training and evalaution using htdra config file.
- Integration of timm model (calling pretraing model and modify the head and tail to run on CIFAR10 with 10 classes).
- Implementation of cog for inferecing the model.
- Buinding a dockerfile to create a docker image.
- Building the docker image using make file.

# Steps to run thorugh the Assignment 

## Run make file to build the docker images
Run makefile using below command to build the docker image atomtically 
```
make build
```
This will build the image with Python version 3.8-slim and image name mullermuttu/emlo2_s2:1.0

To change the base version of the image run below command
```
make build pythonver=3.10
```

## Initiating the training and evalustion using docker image built
Once the docker image built using make file now we can start training by running below command

`docker run -t mullermuttu/emlo2_s2:1.0 python train.py`

We can adjust the below parameters for training using Hydra args in docker run command
* learning_rate: float
* run_eval: true
* max_epochs: int

To run training with learning rate `1e-4` and max_epochs 30:

`docker run -t mullermuttu/emlo2_s2:1.0 python src/train.py learning_rate=1e-4 max_epochs=30`

### Running evaluation
to run the evalution,

`docker run -t mullermuttu/emlo2_s2:1.0 python src/train.py run_eval=true`

## Running the inference using `cog`

To the the inferecnce,

```
cd src/cog
IMAGE_URL=https://gist.githubusercontent.com/bfirsh/3c2115692682ae260932a67d93fd94a8/raw/56b19f53f7643bb6c0b822c410c366c3a6244de2/mystery.jpg
curl $IMAGE_URL > input.jpg
```

`cog predict -i image=@input.jpg`

**cog output**
![cogoutput](\img\cog_output.png)
