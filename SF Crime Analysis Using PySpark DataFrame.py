# Databricks notebook source
# MAGIC %md
# MAGIC ## San Francisco crime data modeling & analysis

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Use PySpark DataFrame to Manipulate Dataset
# MAGIC (https://data.sfgov.org/Public-Safety/sf-data/skgt-fej3/data)
# MAGIC 
# MAGIC Reference: Chicago Crime
# MAGIC https://datascienceplus.com/spark-dataframes-exploring-chicago-crimes/
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC SF Crime Analysis Using PySpark DataFrame:
# MAGIC   
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2266522961450861/3112813335632242/421022585219207/latest.html

# COMMAND ----------

# DBTITLE 1,Import packages
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import to_date, to_timestamp, year, month, dayofmonth, hour, minute
from pyspark.sql.functions import udf, lit
import pyspark.sql.functions as fn
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from ggplot import *
import warnings
import math

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

#download data from SF gov's official website
#import urllib.request
#urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/sf_03_18.csv")
#dbutils.fs.mv("file:/tmp/sf_03_18.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
#display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

#or download the file locally
#https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use pyspark dataframe to manipulate dataset
# MAGIC RDD is registered to the dataframe

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load(data_path)

# COMMAND ----------

df = df.drop(*[s for s in df.columns if s.startswith(":@")]) # drop all the unusful columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.

# COMMAND ----------

# DBTITLE 1,Number of crimes for different category
q1_result = df.groupBy('Category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Conclusions for Q1:
# MAGIC - The most prevelant crime type is Larceny/Theft

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results

# COMMAND ----------

# DBTITLE 1,Number of crimes for different district
q2_result = df.groupBy('PdDistrict').count().orderBy('count', ascending=False)
display(q2_result)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Conclusions for Q2:
# MAGIC - The districts with the most crime count are: Southern, Mission and Northern.
# MAGIC - It would be interesting to consider the population of each district as well.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3 question (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   

# COMMAND ----------

# string to date format: https://stackoverflow.com/a/41273036/7065092
df = df.withColumn("Date", to_date(df.Date, format = "MM/dd/yyyy"))

# COMMAND ----------

# cast coordinate from string to float
df = df \
      .withColumn("Lat", df.Y.cast(FloatType())) \
      .withColumn("Lon", df.X.cast(FloatType()))

# COMMAND ----------

# We define "SF downtown" as 1.5 km within the Montgomery subway station

downtownCenter = (37.784824, -122.407525) # (lat, lon) coordinate of the Montgomery subway station
downtownCenterLat, downtownCenterLon = downtownCenter
downtownRadius = 1.5 # km

# COMMAND ----------

def haversine(lat1, lon1, lat2 = downtownCenterLat, lon2 = downtownCenterLon):
    # calculate distance between two locations
    # given (lat, lon) of two locations, in degree
    # return distance, in km, float type
    lat1, lon1, lat2, lon2 = np.radians((lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(6367 * c)

haversine_udf = udf(haversine, FloatType())  # udf

# COMMAND ----------

df = df.withColumn("DistToCenter", haversine_udf(df.Lat, df.Lon)) # calculate distance to city center

# COMMAND ----------

# DBTITLE 1,Number of crimes in SF downtown, every Sunday
q3_result = df.filter(df.DayOfWeek == "Sunday") \
              .filter(df.DistToCenter < downtownRadius) \
              .groupBy('Date').count() \
              .withColumnRenamed("count", "downtownCount")\
              .orderBy('Date')
display(q3_result)

# COMMAND ----------

# calculate crime count in all area, and join with downtown area count
q3_result = q3_result.join(
                df.filter(df.DayOfWeek == "Sunday") \
                  .groupBy('Date').count() \
                  .withColumnRenamed("count", "allAreaCount"),
                "Date"
              ) 

# calculate the percentage of downtownCrime vs. allCrime, on every Sunday
q3_result = q3_result.withColumn("downtownPercent", q3_result.downtownCount / q3_result.allAreaCount)

# COMMAND ----------

# DBTITLE 1,Percentage of downtown crimes vs. all SF crimes, on every Sunday
display(q3_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### conclusions of Q3
# MAGIC 
# MAGIC - We made two time series: downtown crime every Sunday, and downtown crime percentage (vs. all SF).
# MAGIC - The high spikes are probably related to some special events, such as parade or gathering.
# MAGIC - In the downtown crime **count** plot, there is no obvious long-term trend over time.
# MAGIC - In the downtown crime **percentage** plot, we can see an increaseing trend from 2006 to 2009, and a decreasing trend from 2009 to 2012. 
# MAGIC - If we want to do it very carefully, we can do statistical tests about the trend and slope. Here we're just doing EDA to identify potentially interesting stories.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

# DBTITLE 1,Number of crimes in each month of every year
display(df.groupBy(month(df.Date), year(df.Date)).count())

# COMMAND ----------

# DBTITLE 1,Number of crimes in each month of 2015, 2016 and 2017.
display(df.filter((year(df.Date) < 2018) & (year(df.Date) >= 2015)).groupBy(month(df.Date), year(df.Date)).count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### conclusions of Q4:
# MAGIC 
# MAGIC - From the previous plots, we can see a strong seasonality effect in the monthly crime event time series. 
# MAGIC   - Most significantly, crime event count drops in every Feburary. Speculatively, this might be related to the lower temperature.
# MAGIC 
# MAGIC - For potential visitors, Feburary might be a good time of year to visit, considering the lower frequency of crimes. 
# MAGIC 
# MAGIC - For business owners in SF, consider cautiously lowering security budget for Feburary. 
# MAGIC 
# MAGIC - For police forces and policy makers, it would be useful to find out the reasons behind the seasonality of crime rates, and adjust the policies accordingly. Understanding the reasons behind the seasonality requires comparision with other datasets, such as temperature data and tourism data, and is beyond the scope of this project. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

# convert Time from string to timestamp type
df = df.withColumn("Time", to_timestamp(df.Time, format = "HH:mm"))

# COMMAND ----------

# DBTITLE 1,Number of crimes of each hour, for 3 selected days
display(
  df.filter(
    (df.Date == lit("2015-12-15")) | (df.Date == lit("2016-12-15")) | (df.Date == lit("2017-12-15"))
  ).groupBy(hour(df.Time), df.Date).count()
)

# COMMAND ----------

# DBTITLE 1,Number of crimes of each hour, for all days in our dataset
display(
  df\
  .filter(df.DistToCenter < downtownRadius) \
  .groupBy(hour(df.Time)).count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusion for Q5:
# MAGIC - As can be seen in these two plots, 
# MAGIC   - There are fewer crime events after midnight, and the lowest crime count hours are around 5 am.
# MAGIC   - Hourly crime count is lower in the morning and before noon. Hourly crime count is higher in the afternoon and in the evening.
# MAGIC - Suggestion to visitors:
# MAGIC   - SF is safer in the morning than in the afternoon and evening. 
# MAGIC   - Pay more attention to protect your property and yourself after noon.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger district  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

# DBTITLE 1,Top-3 most dangerous districts: Southern, Mission and Northern
display(df.groupBy("PdDistrict").count().orderBy("count", ascending = False))

# COMMAND ----------

# DBTITLE 1,Total crime count for three most dangerous districts
display(df.groupBy("PdDistrict").count().orderBy("count", ascending = False).limit(3))

# COMMAND ----------

# DBTITLE 1,Find out the most prevalent crime types in the top-3 most dangerous districts
display(
  df \
  .filter((df.PdDistrict == "SOUTHERN") | (df.PdDistrict == "MISSION") | (df.PdDistrict == "NORTHERN")) \
  .groupBy(df.Category) \
  .count() \
  .orderBy("count", ascending = False)
)

# COMMAND ----------

# DBTITLE 1,Hourly crime count for top-3 districts
display(
  df \
  .filter((df.PdDistrict == "SOUTHERN") | (df.PdDistrict == "MISSION") | (df.PdDistrict == "NORTHERN")) \
  .groupBy(hour(df.Time), df.PdDistrict) \
  .count()
)

# COMMAND ----------

# DBTITLE 1,Hourly crime count breakdown for the most dangerous district, Southern District
display(
  df \
  .filter(df.PdDistrict == "SOUTHERN") \
  .groupBy(hour(df.Time), df.Category) \
  .count() \
  .orderBy(hour(df.Time), 'count', df.Category, ascending = [True, False, True])
)

# COMMAND ----------

# DBTITLE 1,Hourly crime count breakdown for the second most dangerous district, Mission District
display(
  df \
  .filter(df.PdDistrict == "MISSION") \
  .groupBy(hour(df.Time), df.Category) \
  .count() \
  .orderBy(hour(df.Time), 'count', df.Category, ascending = [True, False, True])
)

# COMMAND ----------

# DBTITLE 1,Hourly crime count breakdown for the third most dangerous district, Northern District
display(
  df \
  .filter(df.PdDistrict == "NORTHERN") \
  .groupBy(hour(df.Time), df.Category) \
  .count() \
  .orderBy(hour(df.Time), 'count', df.Category, ascending = [True, False, True])
)

# COMMAND ----------

df_q6 = df \
        .groupBy(hour(df.Time), df.PdDistrict, df.Category) \
        .count() \
        .withColumnRenamed("count", "categoryCount") \
        .join(
          df\
            .groupBy(hour(df.Time), df.PdDistrict)\
            .count(),
          on = ["PdDistrict", "hour(Time)"]
         ) \
         .withColumn("categoryPercent", fn.col("categoryCount") / fn.col("count")) 

# COMMAND ----------

# DBTITLE 1,Hourly time series of Theft crime percentage, for top-3 dangerous districts
display(df_q6
        .filter((df_q6.PdDistrict == "SOUTHERN") | (df_q6.PdDistrict == "MISSION") | (df_q6.PdDistrict == "NORTHERN")) \
        .filter(df_q6.Category == "LARCENY/THEFT")
       )

# COMMAND ----------

# DBTITLE 1,Hourly time series of Assault crime percentage, for top-3 dangerous districts
display(df_q6
        .filter((df_q6.PdDistrict == "SOUTHERN") | (df_q6.PdDistrict == "MISSION") | (df_q6.PdDistrict == "NORTHERN")) \
        .filter(df_q6.Category == "ASSAULT")
       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusions for Q6:
# MAGIC - For these three districts, the hourly crime count increases from 5 am, and has two peaks at noon and 6 pm. The hourly crime count significantly decereases from midnight to 5 am. 
# MAGIC - Larceny/Theft is the most prevelant crime category, and its percentage increases continiously from around 4-5 am to 7-8 pm. 
# MAGIC - Theft percentage is much higher in Northern and Southern Districts (as high as ~40% in evenings), than in Mission District (as high as ~20% in evenings). 
# MAGIC - Assault percentage has a peak at around 1-2 am. 
# MAGIC - Suggestions for police force assignment:
# MAGIC   - Overall, police force should be more focused on noon to midnight time, when hourly crime count is higher
# MAGIC   - Theft is the most prevelant crime in these three districts. Police force should be well-prepared to handle theft events and pay more attention to suspecious activities, especially around evening, and in Northern and Southern districts. Consider assigning theft-crime specialized police officers to aforementioned time and districts.
# MAGIC   - Assault event has a spike in percentage at around 1-2 am, when around 10% - 15% of crime events are assults. Police officers should be more alerted and prepared to handle assult cases.
# MAGIC - Similar analysis can be done for other categories as well

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

# DBTITLE 1,First, explore the Resolution types
display(df.groupBy(df.Resolution).count().orderBy("count", ascending = False))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC If and only if the `resolution` is `NONE` or `UNFOUNDED`, then we categorize this event as "not resolved". 

# COMMAND ----------

df_q7 = df.groupBy(df.Category).count().orderBy(df.Category).withColumnRenamed("count", "totalCount")

df_q7 = df_q7.join(
            df.filter((df.Resolution != "NONE") & (df.Resolution != "UNFOUNDED"))\
              .groupBy(df.Category).count()\
              .withColumnRenamed("count", "resolvedCount"), 
           "Category")

df_q7 = df_q7.withColumn("resolvedPercent", df_q7.resolvedCount / df_q7.totalCount)

# COMMAND ----------

# DBTITLE 1,Resolution percentage by category
display(df_q7.orderBy(df_q7.resolvedPercent, ascending = False))

# COMMAND ----------

# DBTITLE 1,Resolution percentage of 10 most prevalent crime categories
display(df_q7.orderBy(df_q7.totalCount, ascending = False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Conclusion for Q7
# MAGIC 
# MAGIC - As can be seen in the plot and table, some very prevelant crime category have very low resolution percentage, especially Larceny/Theft and Vehicle Theft
# MAGIC - Policy makers and police forces should consider methods that might improve the resolution percentage of these two types of crimes. Such as educating the citizens some ways to protect their personal items and vehicles from theft, and installing tracking devices for vehicles. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion/Summary
# MAGIC 
# MAGIC - In order to better understand the crime events in San Francisco, and provide advice to police makers, police forces and visitors, we used Apache Spark distributed computing framework and analyzed over two million rows of crime event data from San Francisco government. 
# MAGIC - We used Spark Dataframe to organize and analyze the data, and performed a series of data visualization and OLAP to analyzed the spatial and temporal distributions of crime events, as well as the categories of the most prevelant crimes and resolution percentage.
# MAGIC - For more detailed conclusions, refer to the plots and conclusions of each question.
# MAGIC - Here are some main points:
# MAGIC   - Temporally, crime counts are lower in colder months (esp. Feburary), and lower after midnight and before dawn (around 5 am).
# MAGIC   - Spatially, crime counts are higher in Southern, Mission and Northern PD districts. Theft accounts for very high percentage of crimes in Southern and Northern districts.
# MAGIC   - The most prevelant crime type, Larceny/Theft, has very significant temporal and spatial distribution patterns. 
# MAGIC   - The resolution rates of some most prevelant crime categories are very low, especially for Theft and Vehicle Theft. 
# MAGIC   - Police forces and policy makers should adjust the security plans and policies based on these information. Visitors should also adjust their travel plans and use cautions as well

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Some Caveats and further study ideas
# MAGIC - Higher crime count could mean actually more crimes, or more police officers on the street that can record these crime events
# MAGIC - Crime count can be adjusted by population in an area
# MAGIC - Data quality of time: police officers tend to round the time to the nearest 5min, 10min, 30min, 1 hour time. It would be interesting to make a model and reconstruct the real temporal distribution of crime events
# MAGIC - Time of crime event actually happening vs. crime being reported or noticed are different. For example, one possible explanation of higher theft rate at evening might be that tourists didn't discover they lost some items until dinner/supper time. Of course, these speculations need to be supported by additional data and further studies

# COMMAND ----------

display(
  df \
  .groupBy(df.Time).count()
)

# COMMAND ----------

# DBTITLE 1,Police officers tend to round time to nearest hour, 30 minutes, 10 minutes, or 5 minutes
display(
  df \
  .groupBy(fn.minute(df.Time)).count() \
  .orderBy("count", ascending = False)
)
