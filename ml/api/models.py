import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

VERSION_ALIAS = [
    'champion'
]

def delete_all_models():
    registered_models = client.search_registered_models()
    for model in registered_models:
        client.delete_registered_model(model.name)

def get_latest_models():
    latest_versions = []

    registered_models = client.search_registered_models()
    for model in registered_models:
        # find latest version of the model.
        for version in model.latest_versions:
            latest_versions.append({
                "name": model.name,
                "version": version.version,
                "run_id": version.run_id
            })

    return latest_versions

def setup_environment_model(run_id, activity_name, version_alias, environment="prod"):
    artifact_path = "model"
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Register the model in the Model Registry specific to the environment
    model_name = f"{environment}.{activity_name}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Get the version of the registered model
    version = model_details.version

    # Set model version tags to denote its status within the environment
    client.set_model_version_tag(name=model_name, version=version, key="validation_status", value="pending")

    # Assign environment-specific alias (e.g., dev_champion, staging_champion, prod_champion)
    alias_name = f"{environment}_{version_alias}"
    client.set_registered_model_alias(name=model_name, alias=alias_name, version=version)

    print(f"Model registered in {environment} environment and assigned alias '{alias_name}'; version: {version}")

def get_models_with_alias(alias, best_version = True):
    models_with_alias = []

    # Fetch all registered models
    registered_models = client.search_registered_models()

    # Iterate through all registered models
    for model in registered_models:
        model_name = model.name
        # Get all versions of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")

        # Check each version for the alias
        print(model_versions)
        all_versions = {}

        for version in model_versions:
            pass

        all_versions = {
            int(version.version) : {
                    'model_name': model_name,
                    'version': version.version,
                    'aliases': version.aliases
                }
            for version in model_versions
            if alias in version.aliases
        }
        print(all_versions)
        if best_version:
            version_options = list(all_versions.keys())

            if version_options:
                best_version_id = max(version_options)
                models_with_alias.append(all_versions[best_version_id])
        else:
            models_with_alias.extend(list(all_versions.keys()))

    return models_with_alias

def get_model_version_by_alias(
        model_name = "prod.your_activity_name",
        alias_name = "prod_champion"):
    model_version = client.get_model_version_by_alias(name=model_name, alias=alias_name)
    return model_version