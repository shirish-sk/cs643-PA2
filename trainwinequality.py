import findspark
findspark.init()
findspark.find()

import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce



# Create spark session
conf = pyspark.SparkConf().setAppName('trainwinequality')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.appName("trainwinequality").getOrCreate() 
#spark = SparkSession.builder.master("local[*]").getOrCreate()


if len(sys.argv) == 2:
	filepn = "/job/"+ str(sys.argv[1])
	data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
	print("***********************************************************************")
	print ("Argument passed is :", str(sys.argv[1]))
	print("***********************************************************************")
else:
	data_test = spark.read.format("csv").load("s3://sskemrs3bucket/ValidationDataset.csv" , header = True ,sep =";" , inferSchema = True)
    data_test.printSchema()
    data_test.show()

# Reading the training dataset
data_train = spark.read.format("csv").load("s3://sskemrs3bucket/TrainingDataset.csv" , header = True ,sep =";", inferSchema = True)

#Clean the CSV file for any quotes if present
old_column_name = data_train.schema.names
print(data_train.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

data_train = reduce(lambda data_train, idx: data_train.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_train)
data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_test)
print(data_train.schema)

# Dropping rows with quality equal to 3 because it contains very little data
data_train_new = data_train.filter(data_train['quality'] != "3")


# Selecting all columns except quality as feature columns from training dataset
feature_cols = [x for x in data_train_new.columns if x != "quality"]

# Using a vector assembler for processing features
vect_assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature")

# Using a standard scaler to remove mean and scale to unit variance for better model analysis
Scaler = StandardScaler().setInputCol('feature').setOutputCol('Scaled_feature')

# Using a logistic regression model for training 
logr = LogisticRegression(labelCol = "quality", featuresCol = "Scaled_feature")

# Creating a pipeline object of stages: [VectorAssmbler, StandardScaler, LogisticRegression]
Pipe = Pipeline(stages=[vect_assembler, Scaler, logr])

# Training the model with train dataset via Pipeline stages
PipeModel = Pipe.fit(data_train_new)

# Saving the Pipeline model results 
PipeModel.write().overwrite().save("s3://sskemrs3bucket/job/Modelfile")

# PGenerating predictions for Train and Validation datasets
try: 
	train_prediction = PipeModel.transform(data_train)
	test_prediction = PipeModel.transform(data_test)
except:
	print("***********************************************************************")
	print ("Please check CSV file :")
	print("***********************************************************************")	

# Creating a evaluator classification object to genertae metrics for predictions
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol = "prediction")


# Calculating the F1 score of the model with Train and Validation datasets
train_F1score = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})

# Calculating the Accuracy of the model with Train and Validation datasets
train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})


# Priting the metrics for the user to see
print("***********************************************************************")
print("++++++++++++++++++++++++++++++ Metrics ++++++++++++++++++++++++++++++++")
print("***********************************************************************")
print("[Train] F1 score = ", train_F1score)
print("[Train] Accuracy = ", train_accuracy)
print("***********************************************************************")
print("[Test] F1 score = ", test_F1score)
print("[Test] Accuracy = ", test_accuracy)
print("***********************************************************************")
