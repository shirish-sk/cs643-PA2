import findspark
findspark.init()
findspark.find()

import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce


# Create a Spark Session 
conf = pyspark.SparkConf().setAppName('analyzewinequality')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.appName("analyzewinequality").getOrCreate()

data_test = spark.read.format("csv").load("/data/ValidationDataset.csv" , header = True ,sep =";" , inferSchema = True)
data_test.printSchema()
data_test.show()

#Clean the CSV file for quotes if present
old_column_name = data_test.schema.names
print(data_test.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), data_test)
print(data_test.schema)




# Create a PipelineModel object to load saved model parameters from Training

try:
    PipeModel = PipelineModel.load("/data/Modelfile")
except:
    print("***********************************************************************")
    print("Model file cannot be found. Please check whether model file is present in the directory of mount\n")
    print("***********************************************************************")
    exit()


# Generate predictions for Input dataset file
try:
    test_prediction = PipeModel.transform(data_test)
except:
    print("***********************************************************************")
    print ("Please check CSV file : labels may be improper")
    print("***********************************************************************")



# Save the resulting predictions with original datset to a CSV File
test_prediction.drop("feature", "Scaled_feature", "rawPrediction", "probability").write.mode("overwrite").option("header", "true").csv("/data/resultdata.csv")

# Creating a evaluator classification object to generate metrics for predictions
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol = "prediction")

# Calculating the F1 score/Accuracy of the model with Test dataset
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

print("***********************************************************************")
print("++++++++++++++++++++++++++++++ Metrics ++++++++++++++++++++++++++++++++")
print("***********************************************************************")
print("[Test] F1 score = ", test_F1score)
print("[Test] F1 score = ", test_accuracy)
print("***********************************************************************")
