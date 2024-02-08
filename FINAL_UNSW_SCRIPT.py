import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Use try-catch
print("Excecution started...")

try:
    # Read the data from the CSV file 
    data_path = r"/home/djagtap_umassd_edu/Payload_data_UNSW.csv"
    #data_path = r"/home/djagtap_umassd_edu/splitted-file-0-0.csv"
    data = pd.read_csv(data_path, encoding="latin1")

    print("File read completed.....")

    # Select relevant columns for clustering
    columns_for_clustering = data.iloc[:, :1500]  
    
    # Perform hierarchical clustering using AgglomerativeClustering
    n_clusters = 10  
    print("Set the number of clusters for UNSW")

    print("Agglomerative Clustering started")

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    print("Agglomerative Clustering completed") 
    print(columns_for_clustering)
    data['cluster_label'] = clustering.fit_predict(columns_for_clustering)
    
    print("Performed Hierarchical clustering completed.....")

    # Display the unique cluster labels
    print("Cluster labels:", data['cluster_label'].unique())
    
    # Count the number of occurrences for each label within clusters
    cluster_label_counts = data.groupby(['cluster_label', 'label']).size().unstack(fill_value=0)
    
    # Display the count for each label within clusters
    print("\nNumber of label within clusters:")
    print(cluster_label_counts)

    # Hierarchical clustering performed using Linkage matrix
    linkage_matrix = linkage(columns_for_clustering.T, method='ward')  
    
    print("Hierarchical clustering performed successfully")

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
