# Databricks notebook source
# MAGIC %md
# MAGIC ## San Francisco crime data modeling & analysis

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Use Spark SQL for big data analysis on SF crime data
# MAGIC (https://data.sfgov.org/Public-Safety/sf-data/skgt-fej3/data)

# COMMAND ----------

# MAGIC %md
# MAGIC ziyizhu74@gmail.com Ellie Zhu
# MAGIC 
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2266522961450861/3112813335632301/421022585219207/latest.html

# COMMAND ----------

# DBTITLE 1,Import package 
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
import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/sf_03_18.csv")
dbutils.fs.mv("file:/tmp/sf_03_18.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"

# COMMAND ----------

# DBTITLE 1,Data preprocessing
#Read data from the data storage
#Split the header by its separator
crime_data_lines = sc.textFile(data_path)
#prepare data: remove "
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])
#get header
header = df_crimes.first()
print(header)

#remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)

#get the total number of data 
print(" The crimes dataframe has {} records".format(crimes.count()))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Solove  big data issues via Spark
# MAGIC approach: use SQL  

# COMMAND ----------

# DBTITLE 1,option 1 to get dataframe and sql
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#Load .csv
df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)

#create temp table
df_opt1.createOrReplaceTempView("sf_crime")

#display first 10 rows
first_ten_rows = spark.sql("SELECT * FROM sf_crime LIMIT 10")
display(first_ten_rows)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1 (OLAP): 
# MAGIC ##### Count the number of crimes for different categories:

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)

#display result
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
##in descending order
crimeCategory_desc = spark.sql("SELECT category, COUNT(*) AS count FROM sf_crime GROUP BY category ORDER BY Count DESC")

#display result
display(crimeCategory_desc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1. Conclusion:
# MAGIC Larceny/Theft category has the highest count, meaning that this type of crime happens the most in SF, whereas trea has the lowest count, trea is observed the least time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2 (OLAP)
# MAGIC Count the number of crimes for different district, and visualize your results:

# COMMAND ----------

##in descending order
crimePdDistrict_desc = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")

#display result
display(crimePdDistrict_desc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2. Conclustion:
# MAGIC Southern PdDistrict has the highest number of crimes, whereas crimes in Richmond are low,
# MAGIC we can conclude that Richmond Park, and Taraval are the top3 safest districts, Southern district is the most dangerous district.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3 (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC Used python user-defined function

# COMMAND ----------

##google search appears SF downtown (financial district)'s coordinates to be: 37.7946° N, 122.3999° W
##therefore, we can assume SF downtown is centered at 37.7946° N, 122.3999° W
##within a range of 1.19 square kilometers (approx. 119ha) is called 'SF downtown' / largest distance to the center point: (1.19/3.14)**0.5 = 0.616 km
##assume 1° ≈ 111 km at longitude & latitude

#obtain longitude & latitude
crimesSFdt_x = spark.sql("SELECT DayOfWeek, Date, Address, X FROM sf_crime")
crimesSFdt_y = spark.sql("SELECT DayOfWeek, Date, Address, Y FROM sf_crime")
#crimesSFdt_x.show()
#crimesSFdt_y.show()

from pyspark.sql.types import FloatType

def x_square_float(x):
    return ((float(x)-(-122.3999))*111)**2 

x_square_udf_float = udf(lambda z: x_square_float(z), FloatType())
crimes3_df = crimesSFdt_x.select('DayOfWeek', 'Date', 'Address', 'X', x_square_udf_float('X').alias('x_float_squared'))
#crimes3_df.show()

#convert to Pandas dataframe
#crimes3_pd_df = crimes3_df.toPandas()
#display(crimes3_pd_df)

def y_square_float(y):
    return ((float(y)-37.7946)*111)**2 

y_square_udf_float = udf(lambda z: y_square_float(z), FloatType())
crimes4_df = crimesSFdt_y.select('DayOfWeek', 'Date', 'Address', 'Y', y_square_udf_float('Y').alias('y_float_squared'))
#crimes4_df.show()

#convert to Pandas dataframe
#crimes4_pd_df = crimes4_df.toPandas()
#display(crimes4_pd_df)

#Create temporary table called df3, df4
crimes3_df.createOrReplaceTempView("df3")
crimes4_df.createOrReplaceTempView("df4")

#write spark.SQL to retrieve distance
crimesSFdt = spark.sql("SELECT dt.Date, COUNT(distance) AS num_of_crimes \
                       FROM ( \
                         SELECT df3.DayOfWeek, df3.Date, df3.Address, SQRT(df3.x_float_squared + df4.y_float_squared) as distance \
                         FROM df3 \
                         INNER JOIN df4 \
                         ON df3.address = df4.address \
                         WHERE df3.DayOfWeek = 'Sunday' \
                       ) AS dt \
                       WHERE dt.distance < 0.616 \
                       GROUP BY dt.Date \
                       ORDER BY num_of_crimes DESC")

#display result
display(crimesSFdt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3. Conclustion:
# MAGIC According to calculations, we can conclude that amongst all Sundays, on 01/01/2006 the number of crimes is at its highest level.
# MAGIC 01/01/2006 appeared to be during New Year Holidays, and this could be a major reason which resulted in a high number of crimes.
# MAGIC Aadditional information needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4 (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for the result?  

# COMMAND ----------

#Spark SQL based - Soulution for Q4
#counts the number of crimes in each month of 2015, 2016, 2017, 2018

#format the date MM/dd/yyyyy to YYYY-MM-DD 
crimeInYearMonth = spark.sql("SELECT year(to_date(date, 'MM/dd/yyyy')) as year, \
                                     month(to_date(date, 'MM/dd/yyyy')) as month, \
                                     COUNT(*) as num_of_crimes \
                              FROM sf_crime \
                              GROUP BY year, month \
                              HAVING year in (2015, 2016, 2017, 2018) \
                              ORDER BY num_of_crimes DESC")
                             
#display result
display(crimeInYearMonth)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4. Conclution:
# MAGIC The number of crimes hit its highest level in March 2015:
# MAGIC looking at the data we obtained from 2015-2018, we can also conclude that the year 2015 has the most crime cases becasue the top-3 months are all in 2015.
# MAGIC 
# MAGIC #### Business impact for the result:
# MAGIC There might be some incidents happened in 2015 (politically or financially).
# MAGIC Therefore, for companies in SF, they might have been impacted to some extent.
# MAGIC In these particular months and years, business activities (in the sector of goods and services) might have been slightly affected as well due to safety concerns expressed by people.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q5 (OLAP)
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give travel suggestion to visit SF. 

# COMMAND ----------

#Spark SQL based - Soulution for Q4
#counts the number of crimes in each month of 2015, 2016, 2017, 2018

crimesDate = spark.sql("SELECT res.date, res.hours, COUNT(*) as num_of_crimes \
                        FROM ( \
                               SELECT date, \
                                      CASE WHEN time BETWEEN '00:00' AND '06:00:00' THEN 'early morning' \
                                           WHEN time BETWEEN '06:00' AND '12:00:00' THEN 'morning' \
                                           WHEN time BETWEEN '12:00' AND '18:00:00' THEN 'afternoon' \
                                           WHEN time BETWEEN '18:00' AND '24:00:00' THEN 'night' \
                                      END AS hours \
                               FROM sf_crime \
                               WHERE date IN ('12/15/2015', '12/15/2016', '12/15/2017') \
                        ) AS res \
                        GROUP BY 1,2 \
                        ORDER BY num_of_crimes DESC")

#display result
display(crimesDate)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5. Conclustion:
# MAGIC Analyzed the number of crime with respect to the hours in 12/15/2015, 12/15/2016; 12/15/2017: between 00:00-6:00 'early morning'; between 6:00-12:00 'morning'; between 12:00-18:00 'afternoon'; and lastly between 18:00-00:00 'night'.
# MAGIC 
# MAGIC #### Travel tips to visit SF:
# MAGIC In general, afternoon and night (between 12:00 and 00:00) have higher crime rates, whereas early morning and morning hours appear to be much safer.
# MAGIC My advice would be if you plan to visit San Francisco, try not to arrive at SF city later than 12PM.
# MAGIC Instead, you should prepare one day earlier and set out in the early morning or in the morning,
# MAGIC and that so after you chcek-in your hotel, you can relax a bit and explore the city during daytimes.
# MAGIC Remember to go back home early (no later than 18:00), because there are more chances to be involed in crimes in SF at night.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Q6 (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

#(1) Step1: Find out the top-3 danger disrict  
crimePdDistrict_desc = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")
display(crimePdDistrict_desc.take(3))

#Top-3 danger district: Southern, Mission, and Norther

# COMMAND ----------

#(2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  

##three districts were calculated respectively:
crimes_in_top3 = spark.sql("SELECT res.category, \
                                   res.PdDistrict, \
                                   res.hours, \
                                   COUNT(*) as num_of_crimes \
                            FROM ( \
                                  SELECT category, PdDistrict, \
                                         CASE WHEN time BETWEEN '00:00' AND '06:00:00' THEN 'early morning' \
                                              WHEN time BETWEEN '06:00' AND '12:00:00' THEN 'morning' \
                                              WHEN time BETWEEN '12:00' AND '18:00:00' THEN 'afternoon' \
                                              WHEN time BETWEEN '18:00' AND '24:00:00' THEN 'night' \
                                         END AS hours \
                                  FROM sf_crime \
                                  WHERE PdDistrict IN ('SOUTHERN', 'MISSION', 'NORTHERN') \
                             ) AS res \
                             GROUP BY 1,2,3 \
                             ORDER BY num_of_crimes DESC")

#display result
display(crimes_in_top3)

# COMMAND ----------

# MAGIC %md
# MAGIC (3) give your advice to distribute the police based on your analysis results. 
# MAGIC from the results we obtained, since we are looking at the top-3 danger districts, police force should be distributed primarily in Southern district, then Nothern district followed by Mission district
# MAGIC police on-duty time is suggested to be at night for LARCENY/THEFT; in the afternoon for OTHER OFFENSES
# MAGIC residents in the city should cooperate with the police and stay at home during dangerous hours from 12:PM to midnight

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Q7 (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give hints to adjust the policy.

# COMMAND ----------

#write subqueries tb1 & tb2 to obtain results where resolution is None("Unresolved") or else("resolved")
#left join two tables together
#calculate percentage of resolution
#cnt_un/(cnt_un+cnt_re) is the percentage of unresolved crimes, saved as cnt_un_pc
#cnt_re/(cnt_un+cnt_re) is the percentage of resolved crimes, saved as cnt_re_pc

crimeCategoryPercentage = spark.sql("SELECT tb3.Category, \
                                            cnt_un/(cnt_un+cnt_re) as cnt_un_pc, \
                                            cnt_re/(cnt_un+cnt_re) as cnt_re_pc \
                                     FROM ( \
                                           SELECT tb1.Category, cnt_un, cnt_re \
                                           FROM( \
                                                SELECT Category, COUNT(resolved_or_not) as cnt_un \
                                                FROM (\
                                                      SELECT Category, \
                                                             CASE WHEN Resolution LIKE 'NONE' THEN 'unresolved' \
                                                             ELSE 'resolved' \
                                                             END AS resolved_or_not \
                                                      FROM sf_crime) \
                                                      WHERE resolved_or_not = 'unresolved' \
                                                      GROUP BY Category \
                                                 ) tb1 \
                                           LEFT JOIN \
                                           (SELECT Category, COUNT(resolved_or_not) as cnt_re \
                                            FROM ( \
                                                  SELECT Category, \
                                                         CASE WHEN Resolution LIKE 'NONE' THEN 'unresolved' \
                                                         ELSE'resolved' \
                                                         END AS resolved_or_not \
                                                  FROM sf_crime) \
                                                  WHERE resolved_or_not = 'resolved' \
                                                  GROUP BY Category \
                                           ) tb2 \
                                           ON tb1.Category = tb2.Category \
                                    ) tb3 \
                                    ORDER BY cnt_un_pc DESC")
 
#display result
display(crimeCategoryPercentage.take(20))


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7. Hints to adjust the policy:
# MAGIC 1) Police officers should focus more on cimes like RECOVERED VEHICLE, VEHICLE THEFT, and LARCENY/THEFT, because this three categories have the highest 'unresolved' percentages.
# MAGIC 
# MAGIC 2) Crimes related to vehicles require additional attention from the police department.
# MAGIC 
# MAGIC 3) Police should take some actions to raise the awareness of vehicle owners, as a result of common vehicles safty/security issues.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Use four sentences to summary your work. Like what you have done, how to do it, what the techinical steps, what is your business impact. 
# MAGIC More details are appreciated. You can think about this a report for your manager. Then, you need to use this experience to prove that you have strong background on big  data analysis.  
# MAGIC Point 1:  what is your story ? and why you do this work ?   
# MAGIC Point 2:  how can you do it ?  keywords: Spark, Spark SQL, Dataframe, Data clean, Data visulization, Data size, clustering, OLAP,   
# MAGIC Point 3:  what do you learn from the data ?  keywords: crime, trend, advising, conclusion, runtime 

# COMMAND ----------

# MAGIC %md
# MAGIC #### HW1 Conclusion:
# MAGIC I have leveraged a large-sized data set of SF crimes using Spark, Spark SQL in Databricks.
# MAGIC The goal was to provide some safety suggestions (travel tips) to residents who live in SF, and to travelers who plan to visit the city.
# MAGIC Prior to jumping into the homework, we need to create a clustering to actualize code implementation.
# MAGIC I did some data cleaning as well as data processing through importing packages and through using lines of SQL code, such as applying map() & filter() functions.
# MAGIC In this homework, I mainly used spark SQL to answer OLAP questions.
# MAGIC Some useful findings include SF top-3 common crimes, SF top-3 danger districts, number of crimes around downtown SF, hours (time) for particular dates, and resolved & unresolved percentages for different crime categories, etc.
# MAGIC 
# MAGIC Advertisings on safety topics are necessarily needed to raise awareness primarily among residents in SOUTHERN, MISSION, and NORTHERN districts.
# MAGIC Furthermore, police officers should focus more on cimes like RECOVERED VEHICLE, VEHICLE THEFT, and LARCENY/THEFT, because this three categories have the highest 'unresolved' percentage.
# MAGIC Runtime for question 3 is relatively long, because the result (dataframe) we want to obtain is large.
# MAGIC 
# MAGIC Overall, my suggestion would be avoiding visiting the city or walking around the streets at night or in the afternoon,
# MAGIC because the chances of being involved in a crime are higher than that during other hours.
# MAGIC In addition, pay extra attention during holidays, because during these time, crimes tend to be higher (for example, see Q3. -> 01/01/2006 Sunday num_of_crimes).
