# Car detection

Car detection app using pretrained model.

## Docker

Create Docker image:
```shell
docker build --rm . -t e-dzia/detectron2:latest
```
Then, run the Docker container in the interactive mode:
```shell
docker run -it -p 8888:8888 e-dzia/detectron2:latest /bin/bash
```

Inside the Docker container, you can run the notebook server:
```shell
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
