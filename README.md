# SPARK
You'll need to use at max Java 17 to run Spark otherwise you'll get various errors.

# AWS
To sync the Test folder to the s3 bucket:
```
aws s3 sync . s3://xavier-project-8-bucket/Test
```

To run the script that was uploaded on S3 connect via ssh to the master node:
```commandline
ssh -i ~/.ssh/emr-cluster-keypair.pem hadoop@ec2-13-38-103-131.eu-west-3.compute.amazonaws.com
```
And then run the following command:
```commandline
spark-submit --deploy-mode cluster s3://xavier-project-8-bucket/feature_extraction.py
```