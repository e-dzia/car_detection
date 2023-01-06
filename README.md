# Car detection

Car detection app using pretrained model.

## Docker

Create Docker image:
```shell
docker build --rm . -t e-dzia/detectron2:latest
```

## Notebook server
To run the notebook server, Dockerfile must be run without the last line:
```
ENTRYPOINT ["python", "/car_detection/app/object_detection.py"]
```

Then, run the Docker container in the interactive mode:
```shell
docker run -it -p 8888:8888 e-dzia/detectron2:latest /bin/bash
```

Inside the Docker container, you can run the notebook server:
```shell
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

## Gradio app

To run the demo application with Detectron2 model usage, the Docker container must be run with additional arguments:
```shell
docker run -it -p 7860:7860 e-dzia/detectron2:latest /bin/bash
```

The app will open on the `http://localhost:7860/`.
It allows choosing from several models and drag&dropping images for prediction.
The result is the image with bounding boxes drawn on top of it and an information about the number of found cars.

![Gradio demo app](images/gradio.gif)
