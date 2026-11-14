Steps:

1. Provision minio using the manifests in minio-aistor directory:

   1. Generate a personal license using the procedure documented in: https://docs.min.io/aistor/operations/licenses/
   2. Update the **secret-minio-aistor-credentials.yaml** file using your recently generated license
   3. Run the following commands:
    ```
    oc apply -f minio-aistor/namespace.yaml
    oc apply -f minio-aistor/
    ```

2.