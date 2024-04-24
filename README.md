# cs643-PA2
# ML Model Training and Dockerization of the pretrained ML model
# Building ML Training model for Wine Quality Prediction in AWS using Apache Spark and EMR

This project is focusing on developing the wine quality prediction ML model using Apache Spark multi node cluster using AWS EMR. The ML model will be trained in parallel on a multinode Apache cluster deployed on the EC2 instances. The resultant trained ML model is further utilized and bundled into a Docker container and the wine predictions are done on a single EC2 instance using this docker container.  

Follwing are the list of AWS cloud platform elements and other tools used for this project
AWS EMR using EC2
Apache  Spark 3.5.0 + Hadoop + Hive
Python 3
Docker
Ubuntu Linux 
AMI Linux

Steps:

Extensively used the AWS EMR workflow to rapidly deploy a 4 node Apache Spark cluster which will be used for distributed ML model Training. (for details of the steps, please refer to my youtube video @ https://youtu.be/TNl3E-ogM_I)
Used python based ML Lib for Apache Spark to trained the ML model on a Training dataset of wine features and quality ratings.
Used the PipelineModel technique to write the trained ML model to the HDFS via the parquet files.
The PipeLineModel technique is again used for loading the trained model into the external prediction applications
A python based prediction application running on a single EC2 instance performs wine quality prediction using the trained model and the PipeLineModel loading technique.
All the model parquet and meatdata file structure and the content is transferred from the S3 to the local EC2 instance where the Docker container is to be built.
All the application required files + trained ML Model are copied and extracted into the Docker and a docker container is created using the buidl descriptor "Dockerfile"   
Finally the wine quality prediction are obtained for the test dataset provided within the container 

# Github code repo: https://github.com/shirish-sk/cs643-PA2
# Docker Hub repo for docker container: ssk45/winequalityservice:latest  (docker pull ssk45/winequalityservice:latest)
# S3 bucket: s3://sskemrs3bucket
