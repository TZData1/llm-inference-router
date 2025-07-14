import os
import sys
import pandas as pd
import argparse
import json
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connect import get_connection

def load_queries(conn, dataset_path, clear_existing=False):
    """Load queries from the preprocessed dataset into the database"""
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    cursor = conn.cursor()
    
    if clear_existing:
        cursor.execute("DELETE FROM queries")
    
    query_data = []
    for _, row in df.iterrows():
        extraction_method = row.get("extraction_method", "raw")
        evaluation_metrics = row.get("evaluation_metrics", "exact_match")
        if "evaluation_metric" in row and "evaluation_metrics" not in row:
            evaluation_metrics = row["evaluation_metric"]
        
        generation_params = {
            "max_new_tokens": int(row.get("max_new_tokens", 8)),
            "stop_sequence": row.get("stop_sequence", "['\n']")
        }
        
        source_id = row.get('id', None)
        
        query_data.append((
            row["processed_input"],
            row["dataset"],
            row["reference"],
            extraction_method,
            evaluation_metrics,
            json.dumps(generation_params),
            source_id
        ))
    
    execute_values(
        cursor,
        """INSERT INTO queries 
           (text, dataset, reference, extraction_method, evaluation_metrics, 
            generation_parameters, source_id) 
           VALUES %s RETURNING query_id""",
        query_data
    )
    
    print(f"Loaded {len(query_data)} queries")
    cursor.close()

def update_queries(conn, dataset_path, start_id, exclude=None):
    """Update existing queries and insert missing ones from a CSV file"""
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
    
    exclude = exclude or []
    df = pd.read_csv(dataset_path)
    cursor = conn.cursor()
    
    # Make sure source_id column exists
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN source_id TEXT")
        print("Added source_id column to queries table")
    except Exception:
        pass  # Column already exists
    
    # Get existing IDs in the specified range
    end_id = start_id + len(df) - 1
    cursor.execute(
        "SELECT query_id FROM queries WHERE query_id BETWEEN %s AND %s ORDER BY query_id",
        (start_id, end_id)
    )
    existing_ids = set(row[0] for row in cursor.fetchall())
    
    # Check for missing IDs
    expected_ids = set(range(start_id, start_id + len(df)))
    missing_ids = expected_ids - existing_ids
    if missing_ids:
        print(f"Found {len(missing_ids)} missing IDs to insert")
    
    updated = 0
    inserted = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        query_id = start_id + i
        source_id = row.get('id', None)  # Get source_id from CSV
        
        if query_id in existing_ids:
            # Update existing entry
            updates = []
            params = []
            
            # Regular fields
            if 'processed_input' not in exclude:
                updates.append("text = %s")
                params.append(row["processed_input"])
            
            if 'reference' not in exclude:
                updates.append("reference = %s")
                params.append(row["reference"])
            
            if 'extraction_method' not in exclude:
                updates.append("extraction_method = %s")
                params.append(row.get("extraction_method", "raw"))
            
            if 'evaluation_metric' not in exclude and 'evaluation_metrics' not in exclude:
                metrics = row.get("evaluation_metrics", row.get("evaluation_metric", "exact_match"))
                updates.append("evaluation_metrics = %s")
                params.append(metrics)
            
            # Source ID
            if source_id and 'id' not in exclude:
                updates.append("source_id = %s")
                params.append(source_id)
            
            # Generation parameters
            if 'max_new_tokens' not in exclude or 'stop_sequence' not in exclude:
                gen_params = {}
                if 'max_new_tokens' not in exclude:
                    gen_params['max_new_tokens'] = int(row.get('max_new_tokens', 8))
                if 'stop_sequence' not in exclude:
                    gen_params['stop_sequence'] = row.get('stop_sequence', "['\n']")
                
                updates.append("generation_parameters = %s")
                params.append(json.dumps(gen_params))
            
            if updates:
                params.append(query_id)
                cursor.execute(f"UPDATE queries SET {', '.join(updates)} WHERE query_id = %s", params)
                updated += 1
        else:
            # Insert new entry with specific ID
            extraction_method = row.get("extraction_method", "raw")
            evaluation_metrics = row.get("evaluation_metrics", "exact_match")
            if "evaluation_metric" in row and "evaluation_metrics" not in row:
                evaluation_metrics = row["evaluation_metric"]
            
            generation_params = {
                "max_new_tokens": int(row.get("max_new_tokens", 8)),
                "stop_sequence": row.get("stop_sequence", "['\n']")
            }
            
            cursor.execute(
                """INSERT INTO queries 
                   (query_id, text, dataset, reference, extraction_method, evaluation_metrics, 
                    generation_parameters, source_id) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    query_id,
                    row["processed_input"], 
                    row["dataset"],
                    row["reference"],
                    extraction_method,
                    evaluation_metrics,
                    json.dumps(generation_params),
                    source_id
                )
            )
            inserted += 1
    
    conn.commit()
    print(f"Updated {updated} queries, inserted {inserted} new queries")
    cursor.close()

def main():
    parser = argparse.ArgumentParser(description="Load or update queries")
    parser.add_argument("--dataset", help="Dataset name for informational purposes or fallback path construction (e.g., 'gsm8k')")
    parser.add_argument("--input-file", help="Path to the specific input CSV file")
    parser.add_argument("--clear", action="store_true", help="Clear existing queries")
    parser.add_argument("--update-from", type=int, help="Starting query ID for updates")
    parser.add_argument("--exclude", nargs='+', help="Fields to exclude from update")
    parser.add_argument("--append-skip-rows", type=int, help="Number of rows to skip in the input CSV when appending")
    parser.add_argument("--append-start-id", type=int, help="The starting query_id for the appended data")
    args = parser.parse_args()
    
    conn = get_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    dataset_path = None # Initialize dataset_path

    try:
        # Determine dataset_path
        if args.input_file:
            dataset_path = args.input_file
            print(f"Using input file: {dataset_path}")
        elif args.dataset:
            # Use dataset name only if input file isn't specified and not in append mode
            if args.append_start_id is None:
                 dataset_path = f"data/preprocessed/{args.dataset}.csv"
                 print(f"Using dataset name to construct path: {dataset_path}")
            else:
                 # In append mode, --input-file is mandatory if --dataset is not also the filename
                 print("Error: --input-file must be provided when using append mode unless dataset name matches filename.")
                 conn.close()
                 return
        else:
             # Allow running without dataset/input file if only clearing? No, path needed.
            print("Error: Either --dataset or --input-file must be provided.")
            conn.close()
            return

        # Check if file exists only if path was determined
        if dataset_path and not os.path.exists(dataset_path):
             print(f"Error: Input file not found: {dataset_path}")
             conn.close()
             return

        # Handle different modes
        if args.append_skip_rows is not None and args.append_start_id is not None:
            if not dataset_path:
                print("Error: --input-file must be specified for append mode.")
                conn.close()
                return

            print(f"Starting append mode: Skipping {args.append_skip_rows} rows, starting DB ID {args.append_start_id}")
            try:
                df = pd.read_csv(dataset_path)
                if args.append_skip_rows >= len(df):
                    print(f"Error: --append-skip-rows ({args.append_skip_rows}) is >= total rows ({len(df)})")
                    conn.close()
                    return

                df_to_append = df.iloc[args.append_skip_rows:]
                print(f"Attempting to append {len(df_to_append)} rows from {dataset_path}")

                cursor = conn.cursor()
                query_data = []
                current_id = args.append_start_id
                for _, row in df_to_append.iterrows():
                    # Extract necessary fields from the row
                    extraction_method = row.get("extraction_method", "raw")
                    evaluation_metrics = row.get("evaluation_metrics", "exact_match")
                    if "evaluation_metric" in row and "evaluation_metrics" not in row:
                        evaluation_metrics = row["evaluation_metric"]

                    generation_params = {
                        "max_new_tokens": int(row.get("max_new_tokens", 8)),
                        "stop_sequence": row.get("stop_sequence", "['\\n']")
                    }
                    # Handle potential NaN or missing values gracefully before json.dumps
                    try:
                        gen_params_json = json.dumps(generation_params)
                    except TypeError as e:
                        print(f"Warning: Could not serialize generation_params for row index {_}: {e}. Skipping row.")
                        # Decide how to handle: skip row, use default, etc. Here we skip.
                        continue # Skip this row


                    source_id = row.get('id', None)
                    # Ensure dataset name comes from CSV if possible, fallback needed?
                    dataset_name_from_csv = row.get('dataset', args.dataset if args.dataset else 'unknown') # Fallback needed


                    # Ensure required fields are present and not NaN
                    processed_input = row.get("processed_input")
                    reference = row.get("reference")

                    if pd.isna(processed_input) or pd.isna(reference) or pd.isna(dataset_name_from_csv):
                         print(f"Warning: Skipping row index {_} due to missing required data (input, reference, or dataset).")
                         continue # Skip row with missing critical data


                    query_data.append((
                        current_id, # Explicitly provide query_id
                        processed_input,
                        dataset_name_from_csv,
                        reference,
                        extraction_method,
                        evaluation_metrics,
                        gen_params_json, # Use the serialized JSON
                        source_id
                    ))
                    current_id += 1

                if not query_data:
                    print("No valid data found to append after skipping and validation.")
                else:
                    execute_values(
                        cursor,
                        """INSERT INTO queries
                           (query_id, text, dataset, reference, extraction_method, evaluation_metrics,
                            generation_parameters, source_id)
                           VALUES %s
                           ON CONFLICT (query_id) DO NOTHING""", # Avoid errors if ID accidentally exists
                        query_data
                    )
                    conn.commit()
                    inserted_count = cursor.rowcount # Get number of rows actually inserted
                    print(f"Append operation complete. Inserted {inserted_count} new queries.")
                    if inserted_count < len(query_data):
                         print(f"Note: {len(query_data) - inserted_count} rows were skipped due to existing query_ids.")

                cursor.close()

            except Exception as e:
                print(f"Error during append operation: {e}")
                conn.rollback() # Rollback on error

        elif args.update_from:
            if not dataset_path:
                print("Error: --input-file or --dataset must be specified for update mode.")
                conn.close()
                return
            # Existing update logic... make sure it uses dataset_path
            print(f"Starting update mode: Updating from DB ID {args.update_from} using {dataset_path}")
            update_queries(conn, dataset_path, args.update_from, args.exclude)

        elif args.clear:
             # Clear needs confirmation or separate handling, current load_queries handles it
             if not dataset_path:
                  print("Warning: --clear specified without --input-file or --dataset. Assuming clear all.")
                  cursor = conn.cursor()
                  cursor.execute("DELETE FROM queries")
                  conn.commit()
                  print("Cleared all queries table.")
                  cursor.close()
             else:
                  # If path specified, load_queries handles the clear flag
                  print(f"Starting load mode with clear: Loading from {dataset_path}")
                  load_queries(conn, dataset_path, clear_existing=True)

        elif dataset_path: # Default action is load if path is valid and no other mode specified
            print(f"Starting load mode: Loading from {dataset_path}")
            load_queries(conn, dataset_path, clear_existing=False) # Default clear=False

        # If no mode was triggered and no path determined, an error was already printed.

    except Exception as e:
         # General exception catch for issues like DB connection
         print(f"An unexpected error occurred: {e}")
         # Ensure connection is closed if it was opened
         if conn:
             try:
                 conn.rollback() # Rollback any potential transaction
             except Exception as db_err:
                 print(f"Error during rollback: {db_err}")

    finally:
        if conn: # Ensure connection is closed
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()