import pandas as pd
import numpy as np
import os
import glob
import joblib
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_and_filter_data(csv_file_path, start_date, end_date):
    """Read CSV file and filter data based on date range."""
    # logging.info(f'Reading and filtering data from {csv_file_path}')
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Date'] >= start_date) & (
        data['Date'] <= end_date)][['Close']]
    date_range = data[(data['Date'] >= start_date) & (
        data['Date'] <= end_date)][['Date']]
    return filtered_data, date_range


def standardize_data(data, stockID, sector, date_range, stock_fig_dir, clusters):
    """Standardize the data."""
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    standardized_data = scaler.fit_transform(data)

    clusterID = clusters[clusters['stockID'] == stockID]['clusterID'].values[0]
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


def prepare_data_for_clustering(sector, start_date, end_date, stock_fig_dir, output_dir):
    """Prepare the data for clustering."""
    logging.info('Preparing data for clustering')
    X = []
    other_cluster = []

    golden_shape = get_golden_length(start_date, end_date, sector)

    csv_file_paths = glob.glob(os.path.join(
        f'stock_data_model/{sector}', '*.csv'))

    if os.path.exists(output_dir):
        clusters = pd.read_csv(output_dir + f'clustering_result_{sector}.csv')

    for csv_file_path in csv_file_paths:
        stockID = os.path.basename(csv_file_path).split('.')[0]
        filtered_data, date_range = read_and_filter_data(
            csv_file_path, start_date, end_date)
        if filtered_data.shape != golden_shape:
            print(filtered_data.shape)
            other_cluster.append(stockID)
            continue

        standardized_data = standardize_data(
            filtered_data, stockID, sector, date_range, stock_fig_dir, clusters)

        data_entry = [stockID] + standardized_data.reshape(-1).tolist()
        X.append(data_entry)
    logging.info(f'Prepared data for {len(X)} stocks')
    return np.array(X, dtype=object), other_cluster


def get_golden_length(start_date, end_date, sector):
    if sector == 'Finance':
        csv_file_path = os.path.join(
            f'stock_data_model/', sector, 'AAME.csv')
    else:
        csv_file_path = os.path.join(
            f'stock_data_model/', sector, 'AAPL.csv')
    filtered_data, _ = read_and_filter_data(
        csv_file_path, start_date, end_date)
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


def save_clustering_results(X, labels, output_file, other_cluster):
    """Save the clustering results to a CSV file."""
    logging.info(f'Saving clustering results to {output_file}')
    df = pd.DataFrame()
    df['stockID'] = X[:, 0]
    df['clusterID'] = labels
    # Adding other_cluster
    other_cluster_df = pd.DataFrame(other_cluster, columns=['stockID'])
    # Assign a clusterID that indicates it's in other_cluster
    other_cluster_df['clusterID'] = -1

    # Concatenate the original dataframe with the other_cluster dataframe
    df = pd.concat([df, other_cluster_df], ignore_index=True)

    df.to_csv(output_file, index=False)


def calculate_start_date(end_date, years_before):
    """Calculate the start date based on the end date and the number of years before."""
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_obj = end_date_obj - timedelta(days=365*years_before)
    return start_date_obj.strftime('%Y-%m-%d')


def main():
    logging.info('Starting clustering process')
    sectors = ['Finance', 'Technology']
    end_date = '2024-06-20'
    years_before = 2  # This can be adjusted based on user input
    start_date = calculate_start_date(end_date, years_before)
    logging.info(f'start_date = {start_date}')

    K = 5
    model_dir = 'model/'
    stock_fig_dir = 'stock_figure/'
    output_dir = 'clustering_results/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(stock_fig_dir):
        os.makedirs(stock_fig_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sector in sectors:
        model_path = os.path.join(model_dir, f'kmeans_model_{sector}.pkl')
        output_file = os.path.join(
            output_dir, f'clustering_result_{sector}.csv')

        X, other_cluster = prepare_data_for_clustering(
            sector, start_date, end_date, stock_fig_dir, output_dir)
        logging.info(f"X.shape = {X.shape}")
        Sum_of_squared_distances = []
        silhouette_avg = []
        K = range(2, 15)
        for num_clusters in K:
            kmeans, labels = perform_clustering(X, num_clusters)
            Sum_of_squared_distances.append(kmeans.inertia_)
            # silhouette score
            silhouette_avg.append(silhouette_score(X[:, 1:], labels))
        plt.plot(K, silhouette_avg, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score')
        plt.title(
            f'Silhouette analysis For Optimal k for {years_before} year data')
        plt.savefig(os.path.join(stock_fig_dir, sector,
                    f'Silhouette analysis_{years_before}.jpg'))
        plt.close()

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Sum of squared distances/Inertia')
        plt.title(f'Elbow Method For Optimal k for {years_before} year data')
        plt.savefig(os.path.join(stock_fig_dir, sector,
                    f'Elbow Method_{years_before}.jpg'))
        plt.close()

        # save_clustering_results(X, labels, output_file, other_cluster)
        # save_model(kmeans, model_path)

        # # Load the model (for testing purposes)
        # loaded_kmeans = load_model(model_path)
        # logging.info('Model loaded successfully')


if __name__ == "__main__":
    main()
