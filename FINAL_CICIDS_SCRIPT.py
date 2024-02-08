import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Use try-catch
try:
    # Read the data from the CSV file 
    data_path = r"/home/djagtap_umassd_edu/Payload_data_CICIDS2017-001.csv"
    #splitted-file-0-0.csv"
    #data_path = r"D:\umaas\git\Thesis\dataset\CIC-IDS2017_DS2\splitted_files_CICIDS\splitted-file-0-0.csv"
    data = pd.read_csv(data_path, encoding="latin1")

    # Select relevant columns for clustering
    columns_for_clustering = data.iloc[:, :1500]  
    
    # Perform hierarchical clustering using AgglomerativeClustering
    n_clusters = 15  # Set the number of clusters for UNSW 
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    data['cluster_label'] = clustering.fit_predict(columns_for_clustering)
    
    # Display the unique cluster labels
    print("Cluster labels:", data['cluster_label'].unique())
    
    # Count the number of occurrences for each label within clusters
    cluster_label_counts = data.groupby(['cluster_label', 'label']).size().unstack(fill_value=0)
    
    # Display the count for each label within clusters
    print("\nNumber of label within clusters:")
    print(cluster_label_counts)

    # Hierarchical clustering performed using Linkage matrix
    linkage_matrix = linkage(columns_for_clustering.T, method='ward')  
    
    # Dendrogram plotted with labels and color coding, label rotation
    #plt.figure(figsize=(20, 10))  # Figure size (Width, Height)
    #dendrogram(linkage_matrix, no_plot=True)  # Prevents automatic plotting
    
    # Save the Dendrogram to a PDF file
    #plt.title('Hierarchical Clustering Dendrogram for UNSW dataset')
    #plt.xlabel('Payload Bytes')
    #plt.ylabel('Distance')
    #plt.savefig('UNSW_dendrogram.pdf')

except ValueError as ve:
    print(f"ValueError: {ve}")
    
except Exception as e:
    print(f"Exception: {e}")
    
except:
    print("Unexpected Exception occurred")
