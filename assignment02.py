from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Start a Spark session
spark = SparkSession.builder.appName("JobPostingsAnalysis").getOrCreate()

# Load the CSV file into a Spark DataFrame
df = spark.read.option("header", "true").option("inferSchema", "true").option("multiLine","true").option("escape", "\"").csv("lightcast_job_postings.csv")

# Register the DataFrame as a temporary SQL view
df.createOrReplaceTempView("job_postings")

# Show Schema and Sample Data
print("---This is Diagnostic check, No need to print it in the final doc---")

# comment the lines below when rendering the submission
df.printSchema()
df.show(5)

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')


from pyspark.sql.functions import col, monotonically_increasing_id

industries_df = df.select(
    col("naics_2022_6"),
    col("naics_2022_6_name"),
    col("soc_5").alias("soc_code"),
    col("soc_5_name").alias("soc_name"),
    col("lot_specialized_occupation_name").alias("specialized_occupation"),
    col("lot_occupation_group").alias("occupation_group")
).distinct().withColumn("industry_id", monotonically_increasing_id())


industries_df = industries_df.select(
    "industry_id",
    "naics_2022_6",
    "naics_2022_6_name",
    "soc_code",
    "soc_name",
    "specialized_occupation",
    "occupation_group"
)

industries_df.show(5, truncate=False)

locations_df = df.select(
    col("location"),
    col("city_name"),
    col("state_name"),
    col("county_name"),
    col("msa"),
    col("msa_name"),
).distinct().withColumn("location_id", monotonically_increasing_id())

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
companies_df = df.select(
    col("company"),
    col("company_name"),
    col("company_raw"),
    col("company_is_staffing"),
).distinct().where(F.col("company").isNotNull()).where(F.col("company_name") != 'Unclassified').withColumn("company_id", monotonically_increasing_id())
company_df = companies_df.dropna()


from pyspark.sql.functions import lower
job_postings_df = df.select(
    col("ID").alias("job_postings_id"),
    "title_clean","employment_type_name","remote_type_name","remote_type","body",
    "min_years_experience","max_years_experience","salary","salary_from","salary_to",
    "posted","expired","duration",
    "company","location","naics_2022_6"
).dropDuplicates(["job_postings_id"]) \
 .join(companies_df.select("company","company_id"), on="company", how="left") \
 .join(locations_df.select("location","location_id"), on="location", how="left") \
 .join(industries_df.select("naics_2022_6","industry_id"), on="naics_2022_6", how="left")


companies_df.createOrReplaceTempView("companies")
industries_df.createOrReplaceTempView("industries")
locations_df.createOrReplaceTempView("locations")
job_postings_df.createOrReplaceTempView("job_postings")



## QUERY 1 Industry-Specific Salary Trends Grouped by Job Titl

tech_salary_trends = spark.sql("""
SELECT
    i.naics_2022_6_name AS industry_name,
    i.specialized_occupation,
    PERCENTILE_APPROX(j.salary, 0.5) AS median_salary
FROM job_postings j
JOIN industries i
    ON j.naics_2022_6 = i.naics_2022_6
WHERE i.naics_2022_6 = 518210
  AND j.salary IS NOT NULL
  AND j.salary > 0
GROUP BY i.naics_2022_6_name, i.specialized_occupation
ORDER BY median_salary DESC
""")


tech_salary_pd = tech_salary_trends.toPandas()

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=tech_salary_pd, x="median_salary", y="specialized_occupation", palette="viridis", hue="specialized_occupation", legend=False)
ax.set_xlabel("Median Salary")
ax.set_ylabel("Specialized Occupation")
ax.set_title("Median Salary by Specialized Occupation (NAICS 518210)")
plt.tight_layout()
plt.savefig('output/tech_salary_trends.png')
plt.show()


# QUERY 2 : Top 5 Companies with the Most Remote Jobs in California

remote_jobs_ca = spark.sql("""
    SELECT
        COALESCE(c.company_name) AS company_name,
        COUNT(*) AS remote_jobs
    FROM job_postings j
    JOIN companies c  ON j.company_id  = c.company_id
    JOIN locations l  ON j.location_id = l.location_id
    WHERE (j.remote_type = 1 OR LOWER(j.remote_type_name) LIKE 'remote%')
      AND l.state_name = 'California'
    GROUP BY COALESCE(c.company_name)
    ORDER BY remote_jobs DESC
    LIMIT 10
""")

remote_jobs_ca_pd = remote_jobs_ca.toPandas()

remote_jobs_ca_pd.head()

plt.figure(figsize=(10, 6))
sns.barplot(x="remote_jobs", y="company_name", data=remote_jobs_ca_pd, palette="viridis", hue="company_name", legend=False)
plt.title("Top 10 Companies Hiring Remote Jobs in California")
plt.xlabel("Number of Remote Jobs")
plt.ylabel("Company")
plt.tight_layout()
plt.savefig('output/remote_jobs_ca.png')
plt.show()


# Query 3 :Monthly Job Posting Trends in California

from pyspark.sql.functions import to_date, year, month

# Add parsed date, year, month
job_postings_with_date = job_postings_df.withColumn(
    "posted_date", to_date("posted", "yyyy-MM-dd")
).withColumn(
    "year", year("posted_date")
).withColumn(
    "month", month("posted_date")
)

job_postings_with_date.createOrReplaceTempView("job_postings_with_date")
locations_df.createOrReplaceTempView("locations")

# SQL Query
monthly_trends_ca = spark.sql("""
    SELECT
        j.year,
        j.month,
        COUNT(*) AS job_count
    FROM job_postings_with_date j
    JOIN locations l
        ON j.location_id = l.location_id
    WHERE l.state_name = 'California'
      AND j.posted_date IS NOT NULL
    GROUP BY j.year, j.month
    ORDER BY j.year, j.month
""")

monthly_trends_ca.show(10)


monthly_trends_ca_pd = monthly_trends_ca.toPandas()

# Convert month numbers to proper labels if needed
monthly_trends_ca_pd["month"] = monthly_trends_ca_pd["month"].astype(int)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_trends_ca_pd, x="month", y="job_count", hue="year", marker='o')
plt.title("Monthly Job Posting Trends in California")
plt.xlabel("Month")
plt.ylabel("Number of Job Postings")
plt.legend(title="Year")
plt.grid(True)
plt.savefig('output/monthly_trends_ca.png')
plt.show()



# Query 4 – Salary Comparisons Across Major US Cities

msa_list = [14460,47900,35620,41860,42660,31080,19100,26420,12420,34980,28140,19740]
salary_comparison_cities = spark.sql(f"""
WITH base AS (
  SELECT
    l.MSA,
    l.MSA_NAME,
    j.SALARY
  FROM job_postings j
  JOIN locations l ON j.LOCATION_ID = l.LOCATION_ID
  WHERE j.SALARY IS NOT NULL AND j.SALARY > 0
    AND l.MSA IN ({",".join(map(str, msa_list))})
),
named AS (
  SELECT
    CASE CAST(MSA AS INT)
      WHEN 14460 THEN 'Boston'
      WHEN 47900 THEN 'Washington DC'
      WHEN 35620 THEN 'New York'
      WHEN 41860 THEN 'San Francisco'
      WHEN 42660 THEN 'Seattle'
      WHEN 31080 THEN 'Los Angeles'
      WHEN 19100 THEN 'Dallas'
      WHEN 26420 THEN 'Houston'
      WHEN 12420 THEN 'Austin'
      WHEN 34980 THEN 'Nashville'
      WHEN 28140 THEN 'Kansas City'
      WHEN 19740 THEN 'Denver'
      ELSE COALESCE(MSA_NAME,'Unknown')
    END AS metro,
    SALARY
  FROM base
)
SELECT
  metro,
  ROUND(AVG(SALARY), 2) AS average_salary,
  COUNT(*) AS job_count
FROM named
GROUP BY metro
ORDER BY average_salary DESC
""")

salary_comparison_cities.show()


salary_comparison_cities_pd = salary_comparison_cities.toPandas()

plt.figure(figsize=(12, 8))
sns.barplot(
    data=salary_comparison_cities_pd,
    x="average_salary",
    y="metro",
    palette="viridis",
    hue="metro",  # Set hue to the y-axis variable
    legend=False  # Hide the legend
)
plt.title("Average Salary Comparison Across Major US Metro Areas")
plt.xlabel("Average Salary")
plt.ylabel("Metro Area")
plt.tight_layout()
plt.savefig('output/salary_comparison_cities.png')
plt.show()
