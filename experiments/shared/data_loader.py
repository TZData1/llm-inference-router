import logging

import pandas as pd

from db.connect import get_connection

logger = logging.getLogger(__name__)


def connect_db():
    """Establish a connection to the database."""
    try:
        conn = get_connection()
        logger.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def load_evaluation_dataset(
    conn, dataset_names: list, samples_per_dataset: int, random_seed: int
):
    """Loads query IDs for evaluation datasets, applying sampling."""
    if not conn:
        logger.error("Database connection is not valid.")
        return pd.DataFrame()

    try:
        query_base = "SELECT query_id, dataset, text FROM queries"
        params = []
        if dataset_names and dataset_names != ["all"]:
            safe_dataset_names = tuple(dataset_names)
            placeholders = ", ".join(["%s"] * len(safe_dataset_names))
            query_base += f" WHERE dataset IN ({placeholders})"
            params.append(safe_dataset_names)
        all_queries_df = pd.read_sql(
            query_base, conn, params=params if params else None
        )
        if all_queries_df.empty:
            logger.warning("No queries found for the specified datasets.")
            return pd.DataFrame()
        sampled_queries_list = []
        for dataset in all_queries_df["dataset"].unique():
            dataset_queries = all_queries_df[all_queries_df["dataset"] == dataset]
            if len(dataset_queries) > samples_per_dataset:
                dataset_samples = dataset_queries.sample(
                    n=samples_per_dataset, random_state=random_seed
                )
            else:
                dataset_samples = dataset_queries
            sampled_queries_list.append(dataset_samples)
        if not sampled_queries_list:
            return pd.DataFrame()

        final_queries_df = pd.concat(sampled_queries_list)
        logger.info(
            f"Loaded and sampled {len(final_queries_df)} query IDs for evaluation."
        )
        return final_queries_df[["query_id", "dataset", "text"]]

    except Exception as e:
        logger.error(f"Error loading evaluation dataset: {e}", exc_info=True)
        return pd.DataFrame()


def check_data_completeness(conn, required_query_ids: list, required_model_ids: list):
    """Checks if pre-generated results exist for all query/model combinations."""
    if not conn:
        logger.error("Database connection is not valid.")
        return False, pd.DataFrame(), pd.DataFrame()
    if not required_query_ids or not required_model_ids:
        logger.warning(
            "Required query IDs or model IDs are empty. Cannot check completeness."
        )
        return True, pd.DataFrame(), pd.DataFrame()

    try:
        query_ids_tuple = tuple(required_query_ids)
        model_ids_tuple = tuple(required_model_ids)
        query = """
            SELECT 
                pr.query_id, 
                pr.model_id, 
                q.dataset, 
                pr.accuracy, 
                pr.energy_consumption, 
                pr.latency, 
                pr.input_tokens, 
                pr.output_tokens 
            FROM pregenerated_results pr
            LEFT JOIN queries q ON pr.query_id = q.query_id
            WHERE pr.query_id IN %s AND pr.model_id IN %s
        """
        params = (query_ids_tuple, model_ids_tuple)
        all_results_df = pd.read_sql(query, conn, params=params)
        logger.info(
            f"Fetched {len(all_results_df)} existing pregenerated results from DB."
        )
        required_df = pd.MultiIndex.from_product(
            [required_query_ids, required_model_ids], names=["query_id", "model_id"]
        ).to_frame(index=False)
        total_required = len(required_df)
        logger.info(f"Total required query/model combinations: {total_required}")

        key_cols = ["query_id", "model_id"]
        if not all(col in all_results_df.columns for col in key_cols):
            logger.error(
                f"Fetched results DataFrame is missing key columns: {key_cols}"
            )
            return False, pd.DataFrame(), pd.DataFrame()

        merged = pd.merge(
            required_df,
            all_results_df[key_cols].drop_duplicates(),
            on=key_cols,
            how="left",
            indicator=True,
        )

        missing_combinations = merged[merged["_merge"] == "left_only"][key_cols]
        is_complete = missing_combinations.empty

        if not is_complete:
            logger.error(
                f"MISSING {len(missing_combinations)} out of {total_required} pre-generated results!"
            )
        else:
            logger.info("Data completeness check passed.")
        return is_complete, missing_combinations, all_results_df

    except Exception as e:
        logger.error(f"Error checking data completeness: {e}", exc_info=True)
        return False, pd.DataFrame(), pd.DataFrame()


def get_model_specs(conn):
    """Loads model specifications from the database."""
    if not conn:
        logger.error("Database connection is not valid.")
        return pd.DataFrame()

    try:
        query = "SELECT model_id, name, parameter_count FROM models"
        models_df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(models_df)} model specifications from DB.")
        return models_df
    except Exception as e:
        logger.error(f"Error loading model specifications: {e}", exc_info=True)
        return pd.DataFrame()


def load_query_features(conn, required_query_ids: list) -> pd.DataFrame:
    """Loads pre-calculated features for a given list of query IDs."""
    if not conn:
        logger.error("Database connection is not valid.")
        return pd.DataFrame()

    if not required_query_ids:
        logger.warning("No query IDs provided. Cannot load features.")
        return pd.DataFrame()

    logger.info(f"Loading features for {len(required_query_ids)} queries...")
    try:
        query_ids_tuple = tuple(required_query_ids)
        query = """
            SELECT 
                query_id, 
                task_type, 
                semantic_cluster, 
                complexity_score
            FROM query_features
            WHERE query_id IN %s
        """
        params = (query_ids_tuple,)
        features_df = pd.read_sql(query, conn, params=params)

        if features_df.empty:
            logger.warning(
                "No features found in query_features table for the requested query IDs."
            )
            return pd.DataFrame()
        features_df = features_df.set_index("query_id")
        missing_feature_ids = set(required_query_ids) - set(features_df.index)
        if missing_feature_ids:
            logger.warning(
                f"Missing features for {len(missing_feature_ids)} query IDs: {list(missing_feature_ids)[:10]}..."
            )

        logger.info(f"Successfully loaded features for {len(features_df)} queries.")
        return features_df

    except Exception as e:
        logger.error(f"Error loading query features: {e}", exc_info=True)
        return pd.DataFrame()
