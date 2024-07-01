import pandas as pd
import numpy as np
import os
import glob
import joblib
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_and_filter_data(csv_file_path, start_date, end_date):
    """Read CSV file and filter data based on date range."""
    # logging.info(f'Reading and filtering data from {csv_file_path}')
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Date'] >= start_date) & (
        data['Date'] <= end_date)][['Open', 'High', 'Low', 'Close']]
    return filtered_data


def standardize_data(data):
    """Standardize the data."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data


def prepare_data_for_clustering(sectors, start_date, end_date):
    """Prepare the data for clustering."""
    logging.info('Preparing data for clustering')
    X = []
    golden_shape = get_golden_length(start_date, end_date)
    for sector in sectors:
        csv_file_paths = glob.glob(os.path.join(
            f'stock_data_model/{sector}', '*.csv'))
        for csv_file_path in csv_file_paths:
            stockID = os.path.basename(csv_file_path).split('.')[0]
            filtered_data = read_and_filter_data(
                csv_file_path, start_date, end_date)
            if filtered_data.shape != golden_shape:
                print(filtered_data.shape)
                continue
            standardized_data = standardize_data(filtered_data)

            data_entry = [stockID] + standardized_data.reshape(-1).tolist()
            X.append(data_entry)
    logging.info(f'Prepared data for {len(X)} stocks')
    return np.array(X, dtype=object)


def get_golden_length(start_date, end_date):
    csv_file_path = os.path.join(
        f'stock_data_model/', 'Finance', 'AAME.csv')
    filtered_data = read_and_filter_data(csv_file_path, start_date, end_date)
    return filtered_data.shape


def perform_clustering(X, n_clusters):
    """Perform KMeans clustering."""
    logging.info(f'Performing KMeans clustering with {n_clusters} clusters')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    print(X.shape)
    kmeans.fit(X[:, 1:])
    labels = kmeans.labels_
    return kmeans, labels


def save_model(model, model_path):
    """Save the KMeans model."""
    logging.info(f'Saving model to {model_path}')
    joblib.dump(model, model_path)


def load_model(model_path):
    """Load the KMeans model."""
    logging.info(f'Loading model from {model_path}')
    return joblib.load(model_path)


def save_clustering_results(X, labels, output_file):
    """Save the clustering results to a CSV file."""
    logging.info(f'Saving clustering results to {output_file}')
    df = pd.DataFrame()
    df['stockID'] = X[:, 0]
    df['clusterID'] = labels
    df.to_csv(output_file, index=False)


def main():
    logging.info('Starting clustering process')
    sectors = ['Finance']
    start_date = '2023-06-20'
    end_date = '2024-06-20'
    K = 5
    model_path = 'model/kmeans_model.pkl'
    output_file = 'clustering_result.csv'

    X = prepare_data_for_clustering(sectors, start_date, end_date)
    kmeans, labels = perform_clustering(X, K)
    save_clustering_results(X, labels, output_file)
    save_model(kmeans, model_path)

    # Load the model (for testing purposes)
    loaded_kmeans = load_model(model_path)
    logging.info('Model loaded successfully')


if __name__ == "__main__":
    main()
