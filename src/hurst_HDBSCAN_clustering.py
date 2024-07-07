import pandas as pd
import numpy as np
import os
import glob
import joblib
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score

from hurst import compute_Hc

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_and_filter_data(csv_file_path, start_date, end_date, golden_shape):
    """Read CSV file and filter data based on date range."""
    # logging.info(f'Reading and filtering data from {csv_file_path}')
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Date'] >= start_date) & (
        data['Date'] <= end_date)][['Close']]

    required_length = golden_shape[0]
    # 检查长度并填补0
    if len(filtered_data) < required_length:
        logging.info(f'filtered_data.shape = {filtered_data.shape}')
        fill_length = required_length - len(filtered_data)
        fill_zeros = pd.DataFrame({'Close': [0] * fill_length})
        filtered_data = pd.concat(
            [fill_zeros, filtered_data], axis=0, ignore_index=True)
    return filtered_data


def standardize_data(data, stockID, sector, stock_fig_dir):
    """Standardize the data."""
    # scaler = StandardScaler()
    scaler = MaxAbsScaler()
    standardized_data = scaler.fit_transform(data)

    # # Plot the 'Close' column
    # plt.figure(figsize=(10, 6))
    # # Column 3 corresponds to 'Close'
    # # print(f"date_range = {date_range}, close = {standardized_data[:, 3]}")
    # # print(date_range.to_numpy().shape)
    # # print(standardized_data[:, 3].shape)
    # plt.plot(date_range.to_numpy().reshape(-1),
    #          standardized_data[:, 3].tolist())
    # plt.xlabel('Date')
    # plt.ylabel('Standardized Close Value')
    # plt.title(f'Standardized Close Prices for {stockID}, cluster: {clusterID}')
    # # Set date format on x-axis to make it less dense
    # plt.gca().xaxis.set_major_locator(
    #     mdates.MonthLocator(interval=3))  # Show every second week
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(stock_fig_dir, sector, f'{stockID}.jpg'))
    # plt.close()
    return standardized_data


def prepare_data_for_clustering(sector, start_date, end_date, stock_fig_dir):
    """Prepare the data for clustering."""
    logging.info('Preparing data for clustering')
    X = []

    golden_shape = get_golden_length(start_date, end_date, sector)

    csv_file_paths = glob.glob(os.path.join(
        f'stock_data_model/{sector}', '*.csv'))

    # if os.path.exists(output_dir):
    #     clusters = pd.read_csv(output_dir + f'clustering_result_{sector}.csv')

    for csv_file_path in csv_file_paths:
        stockID = os.path.basename(csv_file_path).split('.')[0]
        filtered_data = read_and_filter_data(
            csv_file_path, start_date, end_date, golden_shape)
        standardized_data = standardize_data(
            filtered_data, stockID, sector, stock_fig_dir)

        # Evaluate Hurst equation
        try:
            H, c, _ = compute_Hc(standardized_data.reshape(-1),
                                 kind='price', simplified=True)
            data_entry = [stockID] + [H]
            logging.info(f"stockID = {stockID} H = {H}")
            X.append(data_entry)
        except:
            logging.info(f"stockID = {stockID} invalid")

    logging.info(f'Prepared data for {len(X)} stocks')
    return np.array(X, dtype=object)


def get_golden_length(start_date, end_date, sector):
    if sector == 'Finance':
        csv_file_path = os.path.join(
            f'stock_data_model/', sector, 'AAME.csv')
    else:
        csv_file_path = os.path.join(
            f'stock_data_model/', sector, 'AAPL.csv')
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Date'] >= start_date) & (
        data['Date'] <= end_date)][['Close']]
    return filtered_data.shape


def perform_clustering(X):
    """Perform KMeans clustering."""
    # logging.info(f'Performing KMeans clustering with {n_clusters} clusters')
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    hclusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    print(X.shape)
    hclusterer.fit(X[:, 1:])
    labels = hclusterer.labels_
    return labels


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
    df['hurst exponent'] = X[:, 1]

    df.to_csv(output_file, index=False)


def calculate_start_date(end_date, years_before):
    """Calculate the start date based on the end date and the number of years before."""
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_obj = end_date_obj - timedelta(days=int(365*years_before))
    return start_date_obj.strftime('%Y-%m-%d')


def main():
    logging.info('Starting clustering process')
    sectors = ['Finance', 'Technology']
    end_date = '2024-06-20'
    years_before = 0.5  # This can be adjusted based on user input
    start_date = calculate_start_date(end_date, years_before)
    logging.info(f'start_date = {start_date}')

    model_dir = 'model/'
    stock_fig_dir = 'stock_figure_hurst/'
    output_dir = 'clustering_results/'

    os.makedirs(model_dir, exist_ok=True)
    for sector in sectors:
        os.makedirs(os.path.join(stock_fig_dir, sector), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for sector in sectors:
        model_path = os.path.join(
            model_dir, f'hurst_HDBSCAN_model_{sector}.pkl')
        output_file = os.path.join(
            output_dir, f'hurst_clustering_result_{sector}.csv')

        X = prepare_data_for_clustering(
            sector, start_date, end_date, stock_fig_dir)
        logging.info(f"X.shape = {X.shape}")
        labels = perform_clustering(X)
        # Sum_of_squared_distances = []
        # silhouette_avg = []
        # K = range(2, 15)
        # for num_clusters in K:
        #     kmeans, labels = perform_clustering(X, num_clusters)
        #     Sum_of_squared_distances.append(kmeans.inertia_)
        #     # silhouette score
        #     silhouette_avg.append(silhouette_score(X[:, 1:], labels))
        # plt.plot(K, silhouette_avg, 'bx-')
        # plt.xlabel('Values of K')
        # plt.ylabel('Silhouette score')
        # plt.title(
        #     f'Silhouette analysis For Optimal k for {years_before} year data')
        # plt.savefig(os.path.join(stock_fig_dir, sector,
        #             f'Silhouette analysis_{years_before}.jpg'))
        # plt.close()

        # plt.plot(K, Sum_of_squared_distances, 'bx-')
        # plt.xlabel('Values of K')
        # plt.ylabel('Sum of squared distances/Inertia')
        # plt.title(f'Elbow Method For Optimal k for {years_before} year data')
        # plt.savefig(os.path.join(stock_fig_dir, sector,
        #             f'Elbow Method_{years_before}.jpg'))
        # plt.close()

        save_clustering_results(X, labels, output_file)
        # save_model(kmeans, model_path)

        # # Load the model (for testing purposes)
        # loaded_kmeans = load_model(model_path)
        # logging.info('Model loaded successfully')


if __name__ == "__main__":
    main()
