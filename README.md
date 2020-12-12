This project contains code for google landmark recognition Kaggle challenge. The models include efficientnet, resnet50, vgg16, InceptionV3 and enhanced inception models InceptionV100, InceptionV200.

In order to deploy a service please use after setting Kubernetes cluster context:
kubectl create -f deploy.yaml
