import boto3


NUMBER_OF_PRIMARY_NODES = 1
PRIMARY_NODE_INSTANCE_TYPE = 'm5.xlarge'
NUMBER_OF_CORE_NODES = 3
CORE_NODE_INSTANCE_TYPE = 'm5.xlarge'
S3_BUCKET_NAME = 'xavier-project-8-bucket'


if __name__ == '__main__':
    emr_client = boto3.client('emr', region_name='eu-west-3')

    cluster_name = f"project-8-emr-cluster"
    configurations = [
        {
            "Classification": "jupyter-s3-conf",
            "Properties": {
                "s3.persistence.bucket": "xavier-project-8-bucket",
                "s3.persistence.enabled": "true"
            }
        }
    ]

    instance_groups = [
        {
            'Name': 'Primary node',
            'Market': 'ON_DEMAND',
            'InstanceRole': 'MASTER',
            'InstanceType': PRIMARY_NODE_INSTANCE_TYPE,
            'InstanceCount': NUMBER_OF_PRIMARY_NODES,
        },
        {
            'Name': 'Core node',
            'Market': 'ON_DEMAND',
            'InstanceRole': 'CORE',
            'InstanceType': CORE_NODE_INSTANCE_TYPE,
            'InstanceCount': NUMBER_OF_CORE_NODES,
        }
    ]

    cluster_config = {
        'Name': cluster_name,
        'LogUri': f's3://{S3_BUCKET_NAME}/emr-logs/',
        'ReleaseLabel': 'emr-7.3.0',
        'Applications': [
            {'Name': 'Spark'},
            {'Name': 'Hadoop'},
            {'Name': 'JupyterHub'}
        ],
        'Configurations': configurations,
        'BootstrapActions': [
            {
                'Name': 'Install Python packages',
                'ScriptBootstrapAction': {
                    'Path': f's3://{S3_BUCKET_NAME}/bootstrap-emr.sh',
                    'Args': []
                }
            }
        ],
        'Instances': {
            'InstanceGroups': instance_groups,
            'Ec2KeyName': 'emr-cluster-keypair',
            'KeepJobFlowAliveWhenNoSteps': True,
            'TerminationProtected': False,
            'Ec2SubnetId': 'subnet-01050bd392073dd12'
        },
        'VisibleToAllUsers': True,
        'JobFlowRole': 'EMR_EC2_DefaultRole',
        'ServiceRole': 'EMR_DefaultRole'
    }

    try:
        response = emr_client.run_job_flow(**cluster_config)
        cluster_id = response['JobFlowId']
        print(f"EMR cluster '{cluster_name}' is being created with ID: {cluster_id}")

    except Exception as e:
        print(f"Error creating EMR cluster: {str(e)}")
