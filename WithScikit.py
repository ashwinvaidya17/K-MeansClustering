from matplotlib import pyplot as plt


from sklearn.cluster import KMeans

# Dummy data for this example
X = [[5.1,3.5],[4.9,3.0],[4.7,3.2],[4.6,3.1],[5.0,3.6],[5.4,3.9],[4.6,3.4],[5.0,3.4],[4.4,2.9],[4.9,3.1],[5.4,3.7],[4.8,3.4],[4.8,3.0],[4.3,3.0],[5.8,4.0],[5.7,4.4],[5.4,3.9],[5.1,3.5],[5.7,3.8],[5.1,3.8],[5.4,3.4],
[5.1,3.7],[4.6,3.6],[5.1,3.3],[4.8,3.4],[5.0,3.0],[5.0,3.4],[5.2,3.5],[5.2,3.4],[4.7,3.2],[4.8,3.1],[5.4,3.4],[5.2,4.1],[5.5,4.2],[4.9,3.1],[5.0,3.2],[5.5,3.5],
[4.9,3.1],[4.4,3.0]]

model = KMeans(n_clusters=3) #n_clusters creates 3 clusters
model.fit(X)

labels = model.labels_ # stores the class labels for each datapoint. In this example label would be assigned from 0,1,2
centroids = model.cluster_centers_ #to plot each centroid. Not necessary

##Plot 

colors = ['blue', 'green', 'red'] #to color each cluster

for id,element in enumerate(X):
	plt.scatter(element[0], element[1], color=colors[labels[id]])

for element in centroids:
	plt.scatter(element[0], element[1], marker='x', color='black')

plt.show()