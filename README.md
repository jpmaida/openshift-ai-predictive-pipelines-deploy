Python 3.11
OpenShift 3.5.2

Steps:

1. Provision minio using the manifests in minio-aistor directory:

   1. Generate a personal license using the procedure documented in: https://docs.min.io/aistor/operations/licenses/
   2. Update the **secret-minio-aistor-credentials.yaml** file using your recently generated license
   3. Run the following commands:
    ```
    oc apply -f minio-aistor/namespace.yaml
    oc apply -f minio-aistor/
    ```
   4. Wait for all the pods to be in *Running* state: `oc get pods -w -n aistor`
   5. Update the **MINIO_SVC** entry in **.env** file with the service address deployed
   6. Create the **ml-artifacts** bucket
   7. Create inside the bucket the director **datasets** and import the datasets/online-retails.xlsx file

2. Provision OpenShift AI
   1. Start with the subscription.yaml
   2. Apply the others manifests
   
3. Setting and running the pipeline
   1. Apply **high-value-customer-ml-datascienceproject.yaml** file
   2. Update the ml-artifacts-dataconnection.yaml secret with the connection data to MinIO
   3. Update the dashboard-dspa-secret.yaml secret with the connection data to MinIO
   4. Apply all the secrets using the following command: `oc apply -f ml-artifacts-dataconnection.yaml dashboard-dspa-secret.yaml`
   5. Apply the datasciencepipelinesapplication.yaml file: `oc apply -f datasciencepipelinesapplication.yaml`

   NOTE: If you changed where MinIO is deployed (originally the **aistor** namespace), you'll need to change the **objectStorage** configuration.

   6. Generate the pipeline:
      1. Run `python -m venv .venv`
      2. Activate it: `source .venv/bin/activate`
      3. Install the dependencies: `pip install -r requirements.txt`
      4. Run `python pipelines/high_value_customer_pipeline.py`
      5. A YAML file will be generated in the same directory from the python file
   7. Import the YAML file to the pipeline server. Set the name as **high-value-customer-training** and give a nice description

   NOTE: This is a UI action but you can find all the manifests looking for **pipelines.kubeflow.org** group.

   8. Create a run and choose a name for it
   9.  Check for the logs but the expected outcome is Complete/Success
   10. Check the S3, if the pipeline ran successfully, and you'll see that two new directories were created: **models** and **high-value-customer-training**. The first has the pipeline logs and the second the trained model, ready for deploy in the **joblib** format

4. Deploy the model
   1. Path: path to the joblib file
   2. model type: predictive
   3. model location: existing connection
   4. model deployment name: high-value-customer
   5. hardware profile: default
   6. model framework: sklearn - 1
   7. serving runtime template: MLServe ServingRuntime for KServe
   8. Make model deployment available throught external route: checked
   9. Require token auth: unchecked
   10. Add custom runtime env vars:
       1. MLSERVER_MODEL_URI = /mnt/models/best_model.joblib

5. Lets test it
