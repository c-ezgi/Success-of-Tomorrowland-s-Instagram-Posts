#!/usr/bin/env python
# coding: utf-8

# # Documentation for the tomorrowland analysis 
# 
# 1. Setup of environment
# 2. Initial feature exploration
# 3. Timing of posts 
# 4. NLP post content
# 5. NLP post comments
# 6. Machine learning

# ## Setup of the environment

# In[ ]:


### set environment and load libraries needed
spark 
seednumber = 72584

# extra functionality
from pyspark.sql.functions import * 
from pyspark.sql.types import *

# nlp
import re
import string
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer, regexp_tokenize
from textblob import TextBlob
import nltk 
nltk.download('wordnet')

# python data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor as sk_dt
from sklearn.metrics import classification_report, confusion_matrix

# plotting 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# model creation
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.feature import Binarizer, OneHotEncoderEstimator, IndexToString, VectorAssembler, StandardScaler, StringIndexer, VectorIndexer, Bucketizer, Imputer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

# model evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator


# In[ ]:


get_ipython().run_line_magic('sh', 'sudo apt-get install -y graphviz')


# In[ ]:


### Load in datasets on posts, comments and account information

# step 1. Define schema for data (@Lin)
posts_schema = StructType([
  StructField('_c0', IntegerType(), True),
  StructField('post_id', StringType(), True),
  StructField('profile_name', StringType(), True),
  StructField('post_timestamp', TimestampType(), True),
  StructField('post_text', StringType(), True),
  StructField('nbr_pictures', IntegerType(), True),
  StructField('is_multiple_pictures', BooleanType(), True),
  StructField('google_maps_url', StringType(), True),
  StructField('location', StringType(), True),
  StructField('nbr_likes', LongType(), True),
  StructField('nbr_comments', LongType(), True),
  StructField('is_video', StringType(), True),
])

comments_schema = StructType([
  StructField('_c0', IntegerType(), True),
  StructField('comment_id', LongType(), True),
  StructField('commenter_id', LongType(), True),
  StructField('commenter_verified', StringType(), True),
  StructField('post_id', StringType(), True),
  StructField('comment_text', StringType(), True),
  StructField('comment_timestamp', TimestampType(), True),
])

# step 2. Read in all datafiles
posts = spark.read.load(
    'FileStore/tables/tomorrowland_Posts_20*.csv',
    format = 'csv', 
    schema = posts_schema,
    header = True, 
    multiLine = True,
    escape='"',
    sep=';')\
    .drop('_c0')

comments = spark.read.load(
    'FileStore/tables/tomorrowland_Comments_*_rand_sample.csv',
    format = 'csv', 
    schema = comments_schema,
    header = True, 
    multiLine = True,
    escape='"',
    sep=';')\
    .drop('_c0')

### ACCOUNTS INFO COULD BE USED LATER IF INSTAGRAM SCRAPING IS USED
# the last column of the accounts information seems to be a json file. Perhaps useful in the future but for now only
# the id, username, number of followers/follows are the most relevant and easily extracted without touching json info.
# accounts = spark.read.load(
#     'FileStore/tables/project/tomorrowland_Account_info.csv',
#     format = 'csv', 
#     header = True, 
#     sep=';')

posts.show()


# ## Posts dataset - cleaning and feature generation

# In[ ]:


# quick check validity 
posts.describe().show()

# missing value check all columns
posts.select([count(when(col(c).isNull(), c)).alias(c) for c in posts.columns]).show()

# posts_id (indeed unique identifier)
posts.agg(countDistinct('post_id')).show()

# profile name (uninformative)
posts.agg(countDistinct('profile_name')).show()

# picturecount details (93% is just one picture, multiple pictures flag suffices)
posts.groupBy('nbr_pictures').agg(countDistinct('post_id')).show()
posts.groupBy('is_multiple_pictures').agg(countDistinct('post_id')).show()

# location vars (both are same and only available for 2.5% of posts 
# also seems to mostly coincide with posts during festival days, not very interesting variables, drop 
posts.where(col('google_maps_url').isNotNull()).show(49)
posts.groupBy('location').agg(countDistinct('post_id')).show()

# video flag (some erroneous return at end of string, correct and convert to boolean)
posts.groupBy('is_video').agg(countDistinct('post_id')).show()





# In[ ]:


vizdata = posts.select(['nbr_likes', 'nbr_comments']).toPandas()

for i in vizdata.columns:
  fig, ax = plt.subplots()
  sns.distplot(vizdata[i])
  plt.title(i)
  display(fig)
  
fig, ax = plt.subplots()
sns.scatterplot('nbr_likes', 'nbr_comments', data=vizdata)
plt.title('relation likes & comments')
display(fig)


# In[ ]:


quants = [int(i) for i in posts.approxQuantile("nbr_likes", [0.30, 0.60], 0.05)]

posts_clear = posts  .withColumn('is_video', (regexp_replace('is_video', '\r', '') == 'True')              .cast(IntegerType()))  .withColumn('is_multiple_pictures', col('is_multiple_pictures')              .cast(IntegerType()))  .withColumn('likes_level', when(col('nbr_likes') < quants[0], 'low')              .when(col('nbr_likes') < quants[1], 'medium')              .when(col('nbr_likes') >= quants[1], 'high'))  .drop('post_name', 'nbr_pictures', 'google_maps_url', 'location')


display(posts_clear)


# ## Timing of posts

# In[ ]:


weekendsplit = Bucketizer(splits=[1, 6, 7], inputCol='dayofweek', outputCol='is_weekend')
posts_timing = posts_clear  .withColumn('year', year('post_timestamp'))  .withColumn('month', month('post_timestamp'))  .withColumn('dayofweek', dayofweek('post_timestamp'))  .withColumn('hour', hour('post_timestamp'))
posts_timing = weekendsplit.transform(posts_timing)

#display(posts_timing)

df_timing_features = posts_timing.select('post_id', 'year', 'month', 'dayofweek', 'is_weekend')


# In[ ]:


vizdata = posts_timing.toPandas()
for i in ['year', 'month', 'dayofweek', 'hour']:
  fig, ax = plt.subplots()
  sns.lineplot(x=i, y='nbr_likes', data=vizdata)
  plt.title('nbr of likes by ' + i)
  display(fig)
  
for i in ['year', 'month', 'dayofweek', 'hour']:
  fig, ax = plt.subplots()
  sns.lineplot(x=i, y='nbr_comments', data=vizdata)
  plt.title('nbr of comments by ' + i)
  display(fig)


# ## Natural language processing on post_text

# In[ ]:


df_text = posts_clear.select('post_id', 'post_text')  .withColumn('text_no_nbr', regexp_replace(col('post_text'), r'[0-9]{1,}', ''))  .withColumn('text_only_str', regexp_replace(col('text_no_nbr'),  "[{0}]".format(re.escape(string.punctuation)), ''))  .withColumn('has_mention', ((size(split(col("post_text"), r"\@")) - 1) > 0).cast(IntegerType()))  .withColumn('has_hashtag', ((size(split(col("post_text"), r"\#")) - 1) > 0).cast(IntegerType()))
df_text.show(50)


# In[ ]:


# Create function applying stemmer to all elements in list (+ convert to udf for use in pyspark)
def util_stem(in_vec):
    return [PorterStemmer().stem(word) for word in in_vec if len(PorterStemmer().stem(word)) > 2]
stemmer_udf = udf(lambda x: util_stem(x), ArrayType(StringType()))

def util_lemma(list_of_words):
    badtokens = {'rd', 'th', 'h', 'cet'}
    lemma_list = [WordNetLemmatizer().lemmatize(w,'v') for w in list_of_words if w not in badtokens]
    return list(set(lemma_list))
lemma_udf = udf(lambda x: util_lemma(x), ArrayType(StringType()))
n_words = udf(lambda x: len(x), IntegerType())


#Create a Pipeline stage to remove all stopwords from words.
RT = RegexTokenizer(inputCol = "text_only_str", outputCol = "text_words", pattern = "\\W")
SWR = StopWordsRemover(inputCol = 'text_words', outputCol = 'text_words_f')

pipeline_textpreprocessing = Pipeline().setStages([RT, SWR]).fit(df_text)
df_text_2 = pipeline_textpreprocessing.transform(df_text)


# Create new df with vectors containing the stemmed & lemmatized tokens 
df_text_3 = df_text_2  .withColumn("vector_stemmed", stemmer_udf("text_words_f"))  .withColumn('vector_lemma', lemma_udf('text_words_f'))  .withColumn('nbr_words', n_words('vector_lemma'))

display(df_text_3)


# In[ ]:


BOW = CountVectorizer(inputCol = 'vector_lemma', outputCol = 'features_bow')
TF = HashingTF(inputCol = 'vector_lemma', outputCol = 'featuresTF', numFeatures=1000)
IdF = IDF(inputCol = 'featuresTF', outputCol = 'features_idf', minDocFreq=100)
W2V = Word2Vec(inputCol = 'vector_lemma', outputCol = 'features_w2v')

pipeline_textfeatures = Pipeline().setStages([BOW, TF, IdF, W2V]).fit(df_text_3)
df_text_4 = pipeline_textfeatures .transform(df_text_3)

df_text_features = df_text_4.select('post_id', 'has_mention', 'has_hashtag', 'nbr_words', col('features_idf').alias('text_features'))
display(df_text_4)


# In[ ]:


#Count top words in the posts
posts_words = df_text_4    .select('post_text', explode('vector_lemma').alias('post_word'))    .where(length('post_word') > 0)    .select('post_text', trim(lower(col('post_word'))).alias('post_word'))    .select('post_word').groupBy('post_word').agg({"post_word":"count"})    .sort('count(post_word)', ascending= False)
            
posts_words.show(25)


# ## NLP on post comments

# In[ ]:


comments_clean = comments  .withColumn('comment_no_nbr', regexp_replace(col('comment_text'), r'[0-9]{1,}', ''))  .withColumn('comment_str_only', regexp_replace(col('comment_no_nbr'),  "[{0}]".format(re.escape(string.punctuation)), ''))  .withColumn('nbr_mentions', size(split(col("comment_text"), r"\@")) - 1)  .withColumn('is_verified_author', when(col('commenter_verified') == 'True', 1)                                    .when(col('commenter_verified') == 1.0, 1)                                   .when(col('commenter_verified') == 'False', 0)                                    .when(col('commenter_verified') == 0.0, 0)
                                   .when(col('commenter_verified') == 'Not found', 0) \
                                   .when(col('commenter_verified').isNull(), 0) \
                                   .otherwise(col('commenter_verified')))


RT = RegexTokenizer(inputCol = 'comment_str_only', outputCol = 'comment_words', pattern = "\\W")
SWR = StopWordsRemover(inputCol = 'comment_words', outputCol = 'comment_words_f')
pipelineModel = Pipeline().setStages([RT, SWR]).fit(comments)
comments_clean = pipelineModel.transform(comments_clean)
comments_clean.show(50)


# In[ ]:


def getPolarity(t):
  textBlob_review = TextBlob(t)
  return textBlob_review.sentiment[0]
getPolarityUDF = udf(getPolarity, DoubleType())

comments_sent = comments_clean  .withColumn('polarity', getPolarityUDF(col('comment_text'))              .cast(FloatType()))  .withColumn('label', when(col('polarity') > 0.1, 1)              .when(col('polarity') < -0.1, -1)              .otherwise(0))

display(comments_sent)


# In[ ]:


comments_clean = comments_clean.withColumn("comment_words_f", lemma_udf("comment_words_f"))

comments_words = comments_clean    .select('comment_text', explode('comment_words_f').alias('word'))    .where(length('word') > 0)    .select('comment_text', trim(lower(col('word'))).alias('word'))    .select('word').groupBy('word').agg({"word":"count"})    .sort('count(word)', ascending= False)
            
comments_words.show(100)

#Top words in most negative and most positive comments
most_positive_comments = comments_sent.where(col('polarity') >= 0.75)
most_negative_comments = comments_sent.where(col('polarity') <= -0.5)


positive_comments_words = most_positive_comments    .select('comment_text', explode('comment_words_f').alias('p_word'))    .where(length('p_word') > 0)    .select('comment_text', trim(lower(col('p_word'))).alias('p_word'))    .select('p_word').groupBy('p_word').agg({"p_word":"count"})    .sort('count(p_word)', ascending= False)

negative_comments_words = most_negative_comments    .select('comment_text', explode('comment_words_f').alias('n_word'))    .where(length('n_word') > 0)    .select('comment_text', trim(lower(col('n_word'))).alias('n_word'))    .select('n_word').groupBy('n_word').agg({"n_word":"count"})    .sort('count(n_word)', ascending= False)


display(negative_comments_words)
display(positive_comments_words)


# In[ ]:


### declare functions for emoji sentiment analysis
def emoji_ext(a):
  # function for extracting list of emojis from comment
  emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
  return regexp_tokenize(a, emoji)
udf_emoji_ext = udf(emoji_ext, returnType = StringType())

def emoji_dic(x):
  # funtion for counting occurrances of each unique emoji per list
  lt = []
  dic = {}
        
  for j in x:
    if j not in dic:
      t = 1
      dic.update({j : t})
    else:
      t += 1
      dic.update({j: t})
  return dic
udf_emoji_dic = udf(emoji_dic, MapType(StringType(), IntegerType()))

def unique_emoji(x):
    for i in x:
      if x != []:
        return i
udf_unique_emoji = udf(unique_emoji, returnType = StringType())

def emoji_sentiment(x):
  for i in positive_emoji:
    if i in x:
      return 1
  for j in negative_emoji:
    if j in x:
      return -1
  return 0
udf_sentiment_emoji = udf(emoji_sentiment, returnType = IntegerType())


# section 1: extract emojis from text
comments_sent_emoj = comments_sent  .withColumn('emojis', (udf_emoji_ext(col('comment_text'))))  .withColumn('emojis_count', (udf_emoji_dic(col('emojis'))))  .withColumn('unique_emoji', (udf_unique_emoji(col('emojis'))))


# section 2: define positive and negative emotions
topemojis = comments_sent_emoj  .select('unique_emoji')  .groupBy('unique_emoji').agg({"unique_emoji":"count"})  .sort('count(unique_emoji)', ascending= False)

positive_emoji = ['😍','❤','🔥','🙌','😂','♥','👌','😎','👍','🎉','🙏','👏','😁','♡','😘','😻','✌','😊','😏','💙',                 '💕','😉','💜','🙈','💖','💪','💥','😄','💚','💛','😃','😜','💯','💃','😆','😝','💓','💞','🖤',                  '😋','💗','😀','😅','💘','😇','😬','💟','❣','😛','🎊','👻','🙂','👯','😚','🍻','✔','🙃','🕺','✅',                 '😗','😙','😸',':‑)',':)',':-]',':]',':-3',':3',':>',':>','8-)','8)',':-}',':}',':-P',':o)',':c)',':^)','=]','=)',                  ':‑D',':D','8‑D','8D','x‑D','xD','X‑D','XD','=D','=3','B^D',':-))',':\'‑)',':\')\'',':-*',':*',':×\'',';‑)',';)',                  '*-)','*)',';‑]',';]',';^)',':‑,',';D',':‑P',':P','X‑P','XP','x‑p','xp',':‑p',':p',':‑Þ',':Þ',':‑þ',':þ',':‑b',':b','((:'                  'd:','=p','>:P\'','O:‑)','O:)','0:‑3','0:3','0:‑)','0:)','0;^)\'','>:‑)','>:)','}:‑)','}:)','3:‑)','3:)','>;)','>:3','>;3','<3',":')"]

negative_emoji = ['😭','😢','😱','😩','💔','😔','😫','👊','😞','😥','😒','😓','😡','😲','😣','😕','😰','😵','☹',                  '😤','😨','😿','😯','😖','😠','🙁','👎','🙀','😦','😧','😟','😾','👿',':(',':/',':[',':<',':,(']

# section 3: define text sentiment based on emojis and also provide overall combined sentiment for post
comments_sent_emoj = comments_sent_emoj  .withColumn('sentiment_emoji', (udf_sentiment_emoji(col('comment_text'))))  .withColumn('overall_sentiment', when(col('sentiment_emoji') == 1, 1)              .when(col('sentiment_emoji') == -1, -1)              .when(col('label') == -1, -1)              .when(col('label') == 1, 1)              .otherwise(0))
display(comments_sent_emoj)


# In[ ]:


comments_sent_emoj.createOrReplaceTempView('comments_sent_analysis')
df_comments_features = spark.sql(
  ''' SELECT 
        post_id, 
        AVG(overall_sentiment) comments_sentiment, 
        SUM(nbr_mentions) AS comments_nbr_mentions, 
        SUM(is_verified_author) AS comments_nbr_verified 
      FROM 
        comments_sent_analysis 
      WHERE 
        post_id IS NOT NULL 
      GROUP BY 
        post_id 
      ORDER BY 
        post_id
  ''')
  
display(df_comments_features)


# In[ ]:



d = df_comments_features.select('comments_sentiment').toPandas()
sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
f, ax = plt.subplots()

# Plot a historgram and kernel density estimate
sns.distplot(d, color="m")

plt.title('comments sentiment distribution')
display(f)


# In[ ]:


display(df_comments_features)


# ## Machine learning section

# In[ ]:


quant_sent = df_comments_features.approxQuantile('comments_sentiment', [0.50], 0.05)
print(quant_sent)


# In[ ]:


quant_sent = df_comments_features.approxQuantile('comments_sentiment', [0.50], 0.05)
posts_full = posts_clear  .join(df_timing_features, on='post_id', how='left')  .join(df_text_features, on='post_id', how='left')  .join(df_comments_features, on='post_id', how='left')  .withColumn('comments_nbr_mentions', col('comments_nbr_mentions').cast(DoubleType()))  .withColumn('comment_positivity', when(col('comments_sentiment') > quant_sent[0], 'high')             .otherwise('low'))  .withColumn('year', col('year').cast(StringType()))  .withColumn('month', col('month').cast(StringType()))  .withColumn('dayofweek', col('dayofweek').cast(StringType()))
  

pd_posts = posts_full.drop('text_features').toPandas()
display(posts_full)


# In[ ]:


# non ohe features used
features_dt = ['is_multiple_pictures', 'is_video', 'is_weekend', 'has_mention', 'has_hashtag', 'nbr_words', 'comments_nbr_mentions', 'nbr_likes']

# quick one hot encoding 
dummify = pd_posts[['year', 'month', 'dayofweek']].apply(lambda x: x.astype('str'))
X1 = pd.get_dummies(dummify)

alldat = pd.concat([pd_posts[features_dt], X1], axis=1).dropna()
y = alldat['nbr_likes']
X = alldat.drop('nbr_likes', axis=1)

skdt_model = sk_dt(max_depth=3).fit(X, y)
y_pred = skdt_model.predict(X)



from graphviz import Source
from sklearn import tree
graph = Source(tree.export_graphviz(skdt_model, out_file=None, feature_names=X.columns, 
                                    class_names=None, label='all', filled=True, 
                                    leaves_parallel=False, impurity=False, node_ids=False, 
                                    proportion=True, rotate=True, rounded=True, 
                                    special_characters=False, precision=0))
png_bytes = graph.pipe(format='png')

with open('/dbfs/FileStore/dtreg_depth3_3.png','wb') as f:
    f.write(png_bytes)
    


# In[ ]:


data_input = posts_full
stagelist = []

### Step 1.  defining categorical, continuous & outcome feature (comment positivity or nbr likes) to be used, also define type of outcome (categorical or continuous)
catfeat_names_num = ['is_weekend', 'has_mention', 'has_hashtag']
catfeat_names_str = ['year', 'month', 'dayofweek']
contfeat_names = ['nbr_words', 'comments_nbr_mentions']
outcome = 'comment_positivity'
predict_continuous = False

### Step 2. prepare different steps of the pipeline and set up pipeline

# apply stringindexer to categorical features consisting of strings
if len(catfeat_names_str) > 0:
  inputlist = catfeat_names_str
  for name in catfeat_names_str:
    stringindexer = StringIndexer(inputCol = name, outputCol = name + '_ind')
    stagelist += [stringindexer]

# one-hot encode all categorical features
inputnames = [name + '_ind'  for name in catfeat_names_str] + catfeat_names_num
outputnames = [name + '_ohe' for name in catfeat_names_str + catfeat_names_num]
onehotencoder = OneHotEncoderEstimator(inputCols=inputnames, outputCols=outputnames)
stagelist += [onehotencoder]

# imputation missing values
imputer = Imputer(inputCols=['comments_nbr_mentions'], outputCols=['comments_nbr_mentions'])
stagelist += [imputer]

# assemble continuous and categorical vectors
vector_cont = VectorAssembler(inputCols=contfeat_names, outputCol='contfeats')
vector_cat = VectorAssembler(inputCols=[name + '_ohe' for name in catfeat_names_str + catfeat_names_num], outputCol='catfeats')
stagelist += [vector_cont, vector_cat]


if predict_continuous:
  ### Run block only in case of a CONTINUOUS outcome
  pipeline_prepare = Pipeline(stages=stagelist).fit(data_input)

  data_prepared = pipeline_prepare.transform(data_input)    .select('catfeats','contfeats', 'text_features', outcome)    .withColumnRenamed(outcome, 'label')
else:
  ### Run this block only in case of a CATEGORICAL outcome 
  stringIndexerLabel = StringIndexer(inputCol = outcome, outputCol = 'label')
  stagelist += [stringIndexerLabel]

  pipeline_prepare = Pipeline(stages=stagelist).fit(data_input)

  data_prepared = pipeline_prepare.transform(data_input)    .select('catfeats','contfeats', 'text_features', 'label')
  
#Show prepared data
display(data_prepared)


# In[ ]:


### Step 1. Make split
data_train, data_test = data_prepared.randomSplit([0.70, 0.30], seed=seednumber)

print("Number of observations in the training set: %s " % data_train.count())
print("Number of observations in the test set: %s " %data_test.count())


### Step 2. Create new pipeline
scaler = StandardScaler(inputCol = 'contfeats', outputCol = 'scaled_contfeats', withStd = True, withMean = False)
vector_all1 = VectorAssembler(inputCols = ['catfeats', 'scaled_contfeats'], outputCol= 'features')
vector_all2 = VectorAssembler(inputCols = ['features', 'text_features'], outputCol = 'features_wtext')
stagelist = [scaler, vector_all1, vector_all2]

pipeline_split_std = Pipeline(stages = stagelist).fit(data_train)


### Step 3. Apply pipeline to test and train datasets
data_train = pipeline_split_std.transform(data_train)
data_test =  pipeline_split_std.transform(data_test)

data_train_s = data_train.select('features','label')
data_train_nlp = data_train.select(col('features_wtext').alias('features'),'label')
data_test_s = data_test.select('features','label')
data_test_nlp = data_test.select(col('features_wtext').alias('features'),'label')


# In[ ]:


### For continuous outcomes
# Linear regression model
lr = LinearRegression(labelCol    = 'label', 
                      featuresCol = 'features', 
                      maxIter = 40)

lr_pgrid = ParamGridBuilder()  .addGrid(lr.regParam, [0.2, 0.8])  .addGrid(lr.elasticNetParam, [0, 0.5, 1])  .build()

rfreg = RandomForestRegressor(labelCol = 'label',
                      featuresCol = 'features', 
                      minInstancesPerNode = 20,
                      seed = seednumber)

rfreg_pgrid = ParamGridBuilder()  .addGrid(rfreg.maxDepth, [3, 5, 7])  .addGrid(rfreg.maxBins, [32, 100, 200])  .build()

# evaluator
cont_eval = RegressionEvaluator(labelCol      = 'label', 
                                predictionCol = 'prediction', 
                                metricName = 'mse')

### For categorical outcomes
# decision tree
dt = DecisionTreeClassifier(labelCol    = 'label', 
                            featuresCol = 'features')

dt_pgrid = ParamGridBuilder()  .addGrid(dt.maxBins, [32, 80])  .build()

# random forest classifier
rf = RandomForestClassifier(labelCol= 'label', 
                        featuresCol='features', 
                           numTrees = 50)

rf_pgrid = ParamGridBuilder()  .addGrid(rf.maxDepth, [3, 7])  .addGrid(rf.maxBins, [32, 80])  .build()

# naive bayes classifier
nb = NaiveBayes(labelCol = 'label', 
                featuresCol = 'features')

nb_pgrid = ParamGridBuilder()  .addGrid(nb.smoothing, [0, 0.3, 0.8])  .build()

# evaluator
cat_eval = BinaryClassificationEvaluator(labelCol="label", metricName='areaUnderROC')


# In[ ]:


print(NaiveBayes().explainParams())


# In[ ]:


if predict_continuous:
  # linear regression
  lr_cv = CrossValidator(estimator = lr, 
                         estimatorParamMaps = lr_pgrid,
                         evaluator = cont_eval,
                         numFolds = 5) 

  # random forest
  rfreg_cv = CrossValidator(estimator = rfreg, 
                             estimatorParamMaps = rfreg_pgrid,
                             evaluator = cont_eval,
                             numFolds = 5)

  # fit best possible models
  model_lr = lr_cv.fit(data_train_s)
  model_rfreg = rfreg_cv.fit(data_train_s)
  
else:
  # decision tree
  dt_cv = CrossValidator(estimator = dt, 
                             estimatorParamMaps = dt_pgrid,
                             evaluator = cat_eval,
                             numFolds = 5)
  
  # random forest
  rf_cv = CrossValidator(estimator = rf, 
                         estimatorParamMaps = rf_pgrid,
                         evaluator = cat_eval,
                         numFolds = 5)
  
  # naive bayes
  nb_cv = CrossValidator(estimator = nb, 
                         estimatorParamMaps = nb_pgrid,
                         evaluator = cat_eval,
                         numFolds = 5)
  
  # fit best possible models
  model_dt = dt_cv.fit(data_train_s)
  model_rf = rf_cv.fit(data_train_s)
  model_nb = nb_cv.fit(data_train_s)


# In[ ]:


if predict_continuous:
  # best hyperparameters linear regression
  print('Linear regression hyperparameters')
  print('-' * 40)
  print("maxIter =        ", model_lr.bestModel._java_obj.getMaxIter())
  print("regParam =        ", model_lr.bestModel._java_obj.getRegParam())
  print("elasticNetParam = ", model_lr.bestModel._java_obj.getElasticNetParam())
  print('\n')

  # best hyperparameters random forest
  print('Random forest hyperparameters')
  print('-' * 40)
  print("maxDepth =        ", model_rfreg.bestModel._java_obj.getMaxDepth())
  print("maxBins =        ", model_rfreg.bestModel._java_obj.getMaxBins())
  print('\n')
else:
  # best hyperparameters decision tree
  print('Decision tree hyperparameters')
  print('-' * 40)
  print("maxBins =        ", model_dt.bestModel._java_obj.getMaxBins())
  print('\n')
  
  # best hyperparameters random forest
  print('Random forest hyperparameters')
  print('-' * 40)
  print("maxBins =        ", model_rf.bestModel._java_obj.getMaxBins())
  print("maxDepth =        ", model_rf.bestModel._java_obj.getMaxDepth())
  print('\n')
  
  # best hyperparameters naive bayes
  print('Naive bayes hyperparameters')
  print('-' * 40)
  print("smoothing =        ", model_nb.bestModel._java_obj.getSmoothing())
  print('\n')


# In[ ]:


if predict_continuous:
  # make predictions on test set
  predictions_lr = model_lr.transform(data_test_s)
  predictions_rfreg = model_rfreg.transform(data_test_s)

  # display performance 
  print("Linear regression model")
  print('-' * 40)
  print('  R^2  : %g' % cont_eval.evaluate(predictions_lr, {cont_eval.metricName: 'r2'}))
  print('  MAE  : %g' % cont_eval.evaluate(predictions_lr, {cont_eval.metricName: 'mae'}))
  print('  RMSE : %g' % cont_eval.evaluate(predictions_lr, {cont_eval.metricName: 'rmse'}))
  print('\n')

  print("Random forest model")
  print('-' * 40)
  print('  R^2  : %g' % cont_eval.evaluate(predictions_rfreg, {cont_eval.metricName: 'r2'}))
  print('  MAE  : %g' % cont_eval.evaluate(predictions_rfreg, {cont_eval.metricName: 'mae'}))
  print('  RMSE : %g' % cont_eval.evaluate(predictions_rfreg, {cont_eval.metricName: 'rmse'}))
  print('\n')
else:
  predictions_dt = model_dt.transform(data_test_s)
  predictions_rf = model_rf.transform(data_test_s)
  predictions_nb = model_nb.transform(data_test_s)
    
  print("Decision Tree model")
  print('-' * 40)
  print('  AUC  : %g' % cat_eval.evaluate(predictions_dt, {cat_eval.metricName: 'areaUnderROC'}))
  print('  Precision-Recall  : %g' % cat_eval.evaluate(predictions_dt, {cat_eval.metricName: 'areaUnderPR'}))
  print('\n')
  
  print("Random forest model")
  print('-' * 40)
  print('  AUC  : %g' % cat_eval.evaluate(predictions_rf, {cat_eval.metricName: 'areaUnderROC'}))
  print('  Precision-Recall  : %g' % cat_eval.evaluate(predictions_rf, {cat_eval.metricName: 'areaUnderPR'}))
  print('\n')
  
  print("Naive Bayes model")
  print('-' * 40)
  print('  AUC  : %g' % cat_eval.evaluate(predictions_nb, {cat_eval.metricName: 'areaUnderROC'}))
  print('\n')


# In[ ]:


regmodel_best = model_rfreg
clasmodel_best = model_nb

regpred_best = predictions_rfreg.select('label', 'prediction').toPandas()
claspred_best = predictions_nb.select('label', 'prediction').toPandas()


# In[ ]:


regpred_best['sqresid'] = regpred_best['label'] - regpred_best['prediction']

fig, ax = plt.subplots()
plt.axhline(linewidth=2, color='r')
sns.scatterplot(x='label', y ='sqresid', data=regpred_best)
plt.title('Residual plot on Random Forest regression')
plt.xlabel('Nbr of likes on post')
plt.ylim(None, 200000)
plt.xlim(0, 300000)
display(fig)


# In[ ]:


print(classification_report(claspred_best['label'], claspred_best['prediction']))
confmat = confusion_matrix(claspred_best['label'], claspred_best['prediction'])

fig, ax = plt.subplots()

cm = confmat
plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
target_names = ['low positivity', 'high positivity']

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names)
plt.yticks(tick_marks, target_names)


thresh =  cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


#plot_confusion_matrix(cm=confmat, target_names = ['low positivity', 'high positivity'])
display(fig)


# In[ ]:


claspred_best['prediction'].value_counts()

