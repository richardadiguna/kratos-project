# kratos-project

This project is deep learning convolutional neural network model aim to classify required document that submitted by user (lender, borrower). This model classify e-ktp, ktp, dukcapil, others (4 classes). The purpose of the development is to enhance KoinWorks OCR.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development. See deployment for notes on how to deploy the project on a live system.

### Datasets
* The dataset for training the model is gathered with backoffice data.
* The dataset can be found and download here: bit.ly/2SXO5D9

### Prerequisites

1. There are some dependecies you need to install into the environment. Just run requirement.sh in the terminal.

2. GPU must be installed into your hardware before loading the saved model.

3. Pre-trained model of vgg16 download here https://github.com/richardadiguna/image-forgery-detection.git 


### Installing Dependencies

Install all dependecies as easy as run requirement.sh file in this repo.

```bash
.requirement.sh
```

## Deployment

The deployment of this project can be done using docker and tf_serving:gpu-latest image.

## Versioning

Versioning in this project is done manully while generating saved_model.pb file

```bash
python3 serve/ServeModel.py -ver <VERSIOn> -sp <META_FILE_PATH> -sm <PB_FILE_PATH>
```

or

```bash
python3 serve/ServeModel.py -h
```

to see all details about the arguments while running ServeModel.py

## Client
To communicate with docker GRPC, client file use protobuf in tensorflow_serving_apis. You can find the code in clien/Client.py. All RPC communication can only use port 8500 for REST API 8501. In this project we use REST API to communicate.

```python
def get_prediction_from_model(data):
    ...
    return
```

### Visualization
If you training the model the log file will be created (LOG_PATH), you can open all the visualization in tensorboard.

```bash
tensorboard --logdir=LOG_PATH
```

## Authors

* **Richard Adiguna**

## License

This project is licensed under the KoinWorks License.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
