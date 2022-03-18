import os
import sparknlp
from sparknlp import annotator
from sparknlp.annotator import Tokenizer, StopWordsCleaner,\
    LemmatizerModel, BertEmbeddings, ElmoEmbeddings, SentenceEmbeddings, WordEmbeddingsModel
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher

from pyspark.ml import Pipeline
from pyspark.ml import feature as f
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, row_number
from pyspark.sql.types import DoubleType, FloatType, StringType, StructType, StructField
from pyspark.sql.window import Window


class NLPCategory:
    def __init__(self, label, desc_type, train_frac, embed_type, thres, refresh_date, env):
        '''
        1)Init variables
        2)Get related columns based on label 'mch' or 'sdm', get description type 'full' or 'partial'

        ----------
        Parameters
        ----------

        label : str
            'mch' or 'sdm'
        desc_type : str
            'full' or 'partial'
        train_frac : float
            fraction used to in train set
        embed_type : str
            bert, elmo, or word, those are word embedding methods
        thres : float
            threshold for similarity score
        refresh_date : srt
            refresh date for model and saved output
        '''
        self.label = label
        self.desc_type = desc_type
        self.train_frac = train_frac
        self.embed_type = embed_type
        self.thres = thres
        self.refresh_date = refresh_date
        self.env = env

        if self.label == 'mch':
            self.col_label = 'category_L3_mch0'
            self.sub_label_list = ['category_L2_mch1',
                                   'category_L1_mch2', 'category_L0_mch3']
            self.ref = 'sdm_fl_desc'

        elif self.label == 'sdm':
            self.col_label = 'category_L4_sdmfl'
            self.sub_label_list = ['category_sdmcat',
                                   'category_sdmcky', 'category_sdmdept']
            self.ref = 'category_L3_mch0'

        else:
            raise ValueError(
                "Unexpected value of 'label'! It must be either 'mch' or 'sdm'", self.label)

        if self.desc_type == 'full':
            self.desc = 'full_desc'

        elif self.desc_type == 'partial':
            self.desc = 'partial_desc'

        else:
            raise ValueError(
                "Unexpected value of 'desc_type'! It must be either 'full' or 'partial'", self.desc_type)

    def _stratified_sample(self, df):
        '''
        ues stratified sampling method to get test and train from sample_and_test

        '''
        df_interim = df.na.drop(subset=[self.col_label])\
            .groupBy(self.col_label) \
            .count() \
            .orderBy(col("count").desc())\
            .withColumn("fraction", lit(self.train_frac))
        fractions = df_interim.select(
            self.col_label, 'fraction').rdd.collectAsMap()
        df_sampled = df.sampleBy(
            col(self.col_label), fractions=fractions, seed=123)

        return df_sampled

    def _create_train_test_pred_set(self, train_and_test, prediction):
        """return to train, test and prediction set"""

        train = self._stratified_sample(train_and_test)\
            .select('scan_cd',
                    'scan_desc',
                    self.col_label,
                    *self.sub_label_list,
                    self.ref,
                    self.desc)

        # After obtaining train with SampleBy, leftanti join the original df to  have test
        test = train_and_test.join(train, on=['scan_cd'], how='leftanti')\
                             .select('scan_cd',
                                     'scan_desc',
                                     self.col_label,
                                     *self.sub_label_list,
                                     self.ref,
                                     col('{}_test'.format(self.desc)).alias(self.desc))

        train_sup = train_and_test.join(train, on=['scan_cd'], how='leftanti')\
            .select('scan_cd',
                    'scan_desc',
                    self.col_label,
                    *self.sub_label_list,
                    self.ref,
                    self.desc)

        window_spec = Window.partitionBy('scan_desc').orderBy(col('scan_desc'))
        pred = prediction.alias('p')\
            .join(train.alias('t'), how='full')\
            .filter(col('p.{}'.format(self.ref)) == col('t.{}'.format(self.ref)))\
            .select('p.scan_cd', 'p.scan_desc', 'p.{}'.format(self.desc), 'p.{}'.format(self.ref))\
            .withColumn("row_number", row_number().over(window_spec))\
            .filter(col('row_number') == 1)\
            .drop(col('row_number'))

        return train, train_sup, test, pred

    def _model_pipeline(self, norm_p):
        '''
        Define model pipeline

        ----------
        Parameters
        ----------
        norm_p : float
            1.0 or 2.0, use this to normalize the dense vector of full desc
        '''

        col_name = self.desc
        document_assembler = DocumentAssembler()\
            .setInputCol(col_name)\
            .setOutputCol("document")

        tokenizer = Tokenizer()\
            .setInputCols(["document"])\
            .setOutputCol("token")

        normalizer = annotator.Normalizer()\
            .setInputCols(["token"])\
            .setOutputCol("normalized")

        stopwords_cleaner = StopWordsCleaner()\
            .setInputCols("normalized")\
            .setOutputCol("cleanTokens")\
            .setCaseSensitive(False)

        lemma = LemmatizerModel.pretrained('lemma_antbnc')\
            .setInputCols(["cleanTokens"])\
            .setOutputCol("lemma")

        if self.embed_type == 'bert':
            word_embeddings = BertEmbeddings\
                .pretrained('bert_base_cased', 'en')\
                .setInputCols(["document", 'lemma'])\
                .setOutputCol("embeddings")\
                .setCaseSensitive(False)
        elif self.embed_type == 'word':
            word_embeddings = WordEmbeddingsModel().pretrained() \
                .setInputCols(["document", "lemma"]) \
                .setOutputCol("embeddings") \
                .setCaseSensitive(False)
        elif self.embed_type == 'elmo':
            word_embeddings = ElmoEmbeddings\
                .pretrained('elmo', 'en')\
                .setInputCols(["document", 'lemma'])\
                .setOutputCol("embeddings")
        else:
            raise ValueError(
                "Unexpected value of 'embedding_type'! It must be either 'bert','word', or 'elmo'", self.embed_type)

        embeddings_sentence = SentenceEmbeddings()\
            .setInputCols(["document", "embeddings"])\
            .setOutputCol("sentence_embeddings")\
            .setPoolingStrategy("AVERAGE")

        embeddings_finisher = EmbeddingsFinisher()\
            .setInputCols("sentence_embeddings", "embeddings")\
            .setOutputCols("sentence_embeddings_vectors", "embeddings_vectors")\
            .setOutputAsVector(True)\
            .setCleanAnnotations(False)

        explode_vectors = SQLTransformer()\
            .setStatement(f"SELECT EXPLODE(sentence_embeddings_vectors) AS features_{col_name}, * FROM __THIS__")

        vector_normalizer = f.Normalizer() \
            .setInputCol("features_"+col_name) \
            .setOutputCol("normFeatures_"+col_name) \
            .setP(norm_p)

        pipeline = Pipeline()\
            .setStages([
                document_assembler,
                tokenizer,
                normalizer,
                stopwords_cleaner,
                lemma,
                word_embeddings,
                embeddings_sentence,
                embeddings_finisher,
                explode_vectors,
                vector_normalizer])

        return pipeline

    def run_model_pipeline(self, train_and_test, prediction):
        '''
        Run model pipeline and returns embeded dataframes

        ----------
        Parameters
        ----------

        train_and_test : dataframe
            Spark Dataframe, train and test set
        prediction : dataframe
            Spark Dataframe, prediction set

        '''
        pth = "gs://{}-dataproc-staging/nlp/{}".format(self.env, self.label)
        train, train_sup, test, pred = self._create_train_test_pred_set(
            train_and_test, prediction)
        save_date = self.refresh_date.replace('-', '')

        model = self._model_pipeline(2.0).fit(train)
        model.write().overwrite().save(pth + self.embed_type +
                                       self.desc_type + 'Model' + save_date)
        print("Model is saved at " + pth + self.embed_type +
              self.desc_type + 'Model' + save_date)

        test_embeded = model.transform(test)\
            .select(col('scan_cd'),
                    col(self.ref),
                    col('normFeatures_{}'.format(self.desc)),
                    col(self.col_label).alias('label_{}'.format(self.label)))

        col_list = ['scan_cd', 'normFeatures_{}'.format(self.desc)] + \
            [self.ref, self.col_label] + self.sub_label_list

        train_embeded = model.transform(train)\
            .select(col_list)

        train_sup_embeded = model.transform(train_sup)\
            .select(col_list)

        pred_embeded = model.transform(pred)\
            .select(col('scan_cd'),
                    col('scan_desc'),
                    col('normFeatures_{}'.format(self.desc)),
                    col(self.ref))

        return train_embeded, train_sup_embeded, test_embeded, pred_embeded

    def cos_sim_cal_auc_withthres(self, train_embeded, test_embeded):
        '''
        Calculate Accuracy for test set, return auc score

        ----------
        Parameters
        ----------

        train_embeded : dataframe
            Spark Dataframe, trainingset with bert embeded dense vector
        test_embed : dataframe
            Spark Dataframe, testset with bert embeded dense vector
        '''

        cos_sim_udf = F.udf(lambda x, y: float(x.dot(y)), DoubleType())
        cos_sim_res = test_embeded.alias('test').join(train_embeded.alias("train"),
                                                      col('test.{}'.format(self.ref)) == col(
                                                          'train.{}'.format(self.ref)),
                                                      how='inner')\
            .select(
            F.col("test.scan_cd"),
            F.col("train.scan_cd").alias("scan_cd_matched"),
            F.col("label_{}".format(self.label)),
            F.col("train.{}".format(self.col_label)).alias(
                'pred_{}'.format(self.label)),
            cos_sim_udf('test.normFeatures_{}'.format(self.desc),
                        'train.normFeatures_{}'.format(self.desc))
            .alias("similarity_score"))

        if self.label == 'mch':
            print("Similarity Score Threshold Not Implemented")
        elif self.label == 'sdm':
            cos_sim_res = cos_sim_res.filter(
                col("similarity_score") >= self.thres)

        window_spec = Window.partitionBy('scan_cd')\
            .orderBy(col('similarity_score').desc())
        res_df = cos_sim_res.withColumn("row_number", row_number().over(window_spec))\
                            .filter(col('row_number') == 1)\
                            .drop(col('row_number'))

        total = cos_sim_res.select('scan_cd').distinct().count()
        auc_num = res_df.filter(col('pred_{}'.format(self.label))
                                == col('label_{}'.format(self.label)))\
                        .count()

        auc = auc_num/total

        return auc

    def pred_label(self, train_embeded, pred_embeded):
        '''
        Run model pipeline and returns bert_embeded dataframes

        ----------
        Parameters
        ----------

        train_embeded : dataframe
            Spark Dataframe, trainingset with bert embeded dense vector
        test_embed : dataframe
            Spark Dataframe, testset with bert embeded dense vector

        '''
        cos_sim_udf = F.udf(lambda x, y: float(x.dot(y)), DoubleType())
        cos_sim_res = pred_embeded.alias('pred').join(train_embeded.alias("train"),
                                                      col('pred.{}'.format(self.ref)) == col(
                                                          'train.{}'.format(self.ref)),
                                                      how='inner')\
            .select(F.col("pred.scan_cd"),
                    F.col('pred.scan_desc'),
                    F.col("train.scan_cd").alias("scan_cd_matched"),
                    F.col(self.col_label),
                    *self.sub_label_list,
                    cos_sim_udf('pred.normFeatures_{}'.format(self.desc),
                                'train.normFeatures_{}'.format(self.desc))
                    .alias("similarity_score"))

        if self.label == 'mch':
            print("Similarity Score Threshold Not Implemented")
        elif self.label == 'sdm':
            cos_sim_res = cos_sim_res.filter(
                col("similarity_score") >= self.thres)

        return cos_sim_res

    def save_dataset(self, train_and_test, prediction):

        train, train_sup, test, pred = self._create_train_test_pred_set(
            train_and_test, prediction)
        save_date = self.refresh_date.replace('-', '')
        pth = "gs://{}-dataproc-staging/nlp/{}".format(self.env, self.label)

        train.write.format('orc') \
                   .mode("overwrite") \
                   .save(pth + f"{'train' + save_date}")
        print("Train is saved at " + pth + f"{'train' + save_date}")

        train_sup.write.format('orc') \
            .mode("overwrite") \
            .save(pth + f"{'train_sup' + save_date}")
        print("Train is saved at " + pth + f"{'train_sup' + save_date}")

        test.write.format('orc') \
            .mode("overwrite") \
            .save(pth + f"{'test' + save_date}")
        print("Test is saved at " + pth + f"{'test' + save_date}")

        pred.write.format('orc') \
            .mode("overwrite") \
            .save(pth + f"{'pred' + save_date}")
        print("pred is saved at " + pth + f"{'pred' + save_date}")
        return

    def nlp_reassign_category(self, project_name, db_name, spark: sparknlp,
                              cal_auc=False, save=True) -> None:

        # Import train_and_test & prediction
        train_and_test = spark.read.format("bigquery")\
            .option('table', '{}.{}.{}_train_test_nlp'
                    .format(project_name, db_name, self.label))\
            .load()

        prediction = spark.read.format("bigquery")\
            .option('table', '{}.{}.{}_predictionset_nlp'
                    .format(project_name, db_name, self.label))\
            .load()

        train_embeded, train_sup_embeded, test_embeded, pred_embeded = self.run_model_pipeline(
            train_and_test, prediction)
        train_embeded.cache()

        self.save_dataset(train_and_test, prediction)

        if cal_auc is True:
            auc = self.cos_sim_cal_auc_withthres(train_embeded, test_embeded)
            print("auc score is {}%".format(round(auc*100, 4)))
            data = [{"run_model_type": self.label,
                     "desc_type": self.desc_type,
                     "embed_type": self.embed_type,
                     "test_auc": float(auc),
                     "refresh_date": self.refresh_date}]

            schema = StructType([StructField('run_model_type', StringType(), True),
                                 StructField('desc_type', StringType(), True),
                                 StructField('embed_type', StringType(), True),
                                 StructField('test_auc', FloatType(), True),
                                 StructField('refresh_date', StringType(), True)])

            output_df = spark.createDataFrame(data, schema)
            output_df.write.format('bigquery') \
                .mode("append") \
                .option("table",
                        "{}.{}.nlp_training_output_{}".format(project_name, db_name, self.label))\
                .save()

        elif cal_auc is False:
            print("SKIP AUC CALCULATION")
        else:
            raise ValueError(
                "Unexpected value of 'cal_auc'! It must be either 'True' or 'False'", cal_auc)

        if save is True:
            save_date = self.refresh_date.replace('-', '')
            pth = "gs://{}-dataproc-staging/nlp/{}".format(self.env, self.label)

            train_embeded_all = train_embeded.union(train_sup_embeded)

            pred_res = self.pred_label(train_embeded_all, pred_embeded)
            pred_res.cache()
            pred_res.write.format('bigquery') \
                .mode("overwrite") \
                .option("table",
                        "{}.{}.cos_sim_res_{}".format(project_name, db_name, self.label))\
                .save()
            print("Cosine Similarity Result Table is written into ",
                  "{}.{}.cos_sim_res_{}".format(project_name, db_name, self.label))

            pred_res.write.format('orc') \
                    .mode("overwrite") \
                    .save(pth + f"{'cos_sim_res_'+ self.label + save_date}")
            print("Cosine Similarity Result Table is saved at " +
                  pth + f"{'cos_sim_res' + save_date}")
        elif save is False:
            print("SKIP SAVE COSINE SIMILARITY RESULT TABLE")
        else:
            raise ValueError(
                "Unexpected value of 'save'! It must be either 'True' or 'False'", save)

        return
