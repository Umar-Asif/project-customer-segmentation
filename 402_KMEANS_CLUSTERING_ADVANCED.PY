###################################################################
## K MEANS CLUSTERING   - 
###################################################################


# IMPORT REQUIRED PYTHON PACKAGES



from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



###################################################################
## CREATE THE DATA
###################################################################

# IMPORT TABLE

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")


# MERGE ON PRODUCT AREA NAME

transactions = pd. merge(transactions, product_areas, how = "inner", on = "product_area_id")

# DROP THE NON-FOOD CATEGORY

transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# AGREGATE SALES AT CUSTOMER LEVEL(BY PRODUCT AREA)

transaction_summary= transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# PIVOT DATA TO PLACE PRODUCT AREA AS COLUMNS


transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",
                                                    aggfunc = "sum",
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None,axis = 1)

# TURN SALES IN TO % SALES



transaction_summary_pivot = transaction_summary_pivot .div(transaction_summary_pivot["Total"], axis = 0 )




# DROP THE TOTAL COLUMN

data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)




###################################################################
## DATA PREPARATION & CLEANING
###################################################################


# CHECK FOR MISSING VALUES


data_for_clustering.isna().sum()


# NORMALIZE DATA


scale_norm = MinMaxScaler()


data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns )


###################################################################
## USE WCSS TO FIND A GOOD VALUE FOR k
###################################################################


k_values = list (range(1,10))
wcss_list = []

for k in  k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)



plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()




###################################################################
## INSTANTIATE AND FIT MODEL
###################################################################


kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(data_for_clustering_scaled)


###################################################################
## USE CLUSTER INFORMATION
###################################################################



# ADD CLUSTER LABEL TO OUR DATA



data_for_clustering["cluster"] = kmeans.labels_



# CHECK CLUSTER SIZE


data_for_clustering["cluster"].value_counts()



###################################################################
## PROFILE OUR CLUSTERS
###################################################################


cluster_summary = data_for_clustering.groupby("cluster")[["Dairy","Fruit", "Meat", "Vegetables"]].mean().reset_index()


















