import os
import subprocess  # nosec
import sys
import datetime
import argparse


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # nosec


install('spark-nlp==3.3.0')

try:
    from nlp_recat_class import NLPCategory

except ImportError:
    from nlp.functions.nlp_recat_class import NLPCategory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str,
                        required=True, dest='project_name')
    parser.add_argument("--db_name", type=str, required=True, dest='db_name')
    parser.add_argument("--staging_bucket", type=str,
                        required=True, dest='bucket')
    parser.add_argument("--env", type=str, required=True, dest='env')
    args = parser.parse_args()
    # pylint: disable=W0401
    import sparknlp

    spark = sparknlp.start()
    bucket = args.bucket
    spark.conf.set('temporaryGcsBucket', bucket)
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', -1)
    spark.conf.set('spark.sql.debug.maxToStringFields', 250)
    spark.conf.set('spark.sql.broadcastTimeout', 1000)
    print('start')

    project_name = args.project_name
    db_name = args.db_name
    env = args.env

    # variables to initialize NLPCatgeory
    label = 'sdm'
    desc_type = 'partial'
    embed_type = 'elmo'
    train_frac = 0.9
    thres = 0.90
    refresh_date = datetime.datetime.today().date().strftime("%Y-%m-%d")

    # Initialize
    sdmnlp = NLPCategory(label, desc_type, train_frac,
                         embed_type, thres, refresh_date, env)
    # Call nlp_reassign_category function
    sdmnlp.nlp_reassign_category(project_name, db_name,
                                 spark, cal_auc=True, save=True)
