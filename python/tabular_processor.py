import argparse
import pandas as pd
import json
import os
import sys

def get_preview(df):
    """Returns a dictionary representation of the dataframe preview."""
    # Replace NaN with None (which becomes null in JSON)
    df_preview = df.head(10).replace({float('nan'): None})
    
    return {
        "columns": list(df.columns),
        "data": df_preview.values.tolist(),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict()
    }

def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

def main():
    parser = argparse.ArgumentParser(description="Tabular Data Processor")
    parser.add_argument("--action", type=str, required=True, choices=['load', 'process'])
    parser.add_argument("--file", type=str, required=True, help="Path to input file")
    parser.add_argument("--out", type=str, help="Path to save processed file")
    parser.add_argument("--params", type=str, help="JSON string of parameters for processing")

    args = parser.parse_args()

    try:
        if args.action == 'load':
            df = load_data(args.file)
            result = get_preview(df)
            result['status'] = 'success'
            result['loaded_path'] = args.file
            print(json.dumps(result))

        elif args.action == 'process':
            df = load_data(args.file)
            params = json.loads(args.params)
            op = params.get('operation')

            if op == 'drop_missing':
                df.dropna(inplace=True)
            
            elif op == 'fill_missing':
                method = params.get('method', 'mean')
                # Simple implementation for numeric only usually, but let's try broadly
                if method == 'mean':
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif method == 'median':
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif method == 'mode':
                    for col in df.columns:
                        df[col] = df[col].fillna(df[col].mode()[0])
                elif method == 'zero':
                     df.fillna(0, inplace=True)

            elif op == 'label_encode':
                from sklearn.preprocessing import LabelEncoder
                cols = params.get('columns', [])
                le = LabelEncoder()
                for col in cols:
                    if col in df.columns:
                         # Ensure type is string for consistent encoding
                         df[col] = le.fit_transform(df[col].astype(str))
            
            elif op == 'one_hot_encode':
                cols = params.get('columns', [])
                if cols:
                    df = pd.get_dummies(df, columns=cols, drop_first=params.get('drop_first', False))

            # Save the result
            save_path = args.out if args.out else args.file
            if save_path.endswith('.csv'):
                df.to_csv(save_path, index=False)
            else:
                df.to_excel(save_path, index=False)

            result = get_preview(df)
            result['status'] = 'success'
            result['message'] = f"Operation {op} completed."
            result['file_path'] = save_path
            print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    main()
