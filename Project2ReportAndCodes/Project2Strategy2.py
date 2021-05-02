
# coding: utf-8

# In[48]:


from Precode2 import *
import numpy
data = np.load('AllSamples.npy')


# In[49]:


k1,i_point1,k2,i_point2 = initial_S2('1417') # please replace 0111 with your last four digit of your ID


# In[50]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[51]:


i_point1


# In[52]:


# Get initial Centroids for k1
import pandas as pd
# get 2nd centroid
data_centroid1 = pd.DataFrame(data, columns = ['x', 'y'])
data_centroid1['distance_from_1'] = numpy.sqrt((data_centroid1['x'] - i_point1[0])**2 + (data_centroid1['y'] - i_point1[1])**2)
k1_2 = numpy.argmax(data_centroid1['distance_from_1'])
k1_2_list = [data_centroid1['x'][k1_2], data_centroid1['y'][k1_2]]
i_point1_list = numpy.zeros((4, 2))
i_point1_list[0] = i_point1
i_point1_list[1] = k1_2_list
data_centroid1 = data_centroid1.drop([k1_2])

# get 3rd centroid
data_centroid1['distance_from_2'] = numpy.sqrt((data_centroid1['x'] - k1_2_list[0])**2 + (data_centroid1['y'] - k1_2_list[1])**2)
data_centroid1['distance_from_12'] = (data_centroid1['distance_from_2'] + data_centroid1['distance_from_1']) / 2
k1_3 = numpy.argmax(data_centroid1['distance_from_12'])
k1_3_list = [data_centroid1['x'][k1_3], data_centroid1['y'][k1_3]]
i_point1_list[2] = k1_3_list
data_centroid1 = data_centroid1.drop([k1_3])
                                     

# get 4th centroid
data_centroid1['distance_from_3'] = numpy.sqrt((data_centroid1['x'] - k1_3_list[0])**2 + (data_centroid1['y'] - k1_3_list[1])**2)
data_centroid1['distance_from_123'] = (data_centroid1['distance_from_2'] + data_centroid1['distance_from_1'] + data_centroid1['distance_from_3']) / 3
k1_4 = numpy.argmax(data_centroid1['distance_from_123'])
k1_4_list = [data_centroid1['x'][k1_4], data_centroid1['y'][k1_4]]
i_point1_list[3] = k1_4_list

    
    
i_point1 = i_point1_list
i_point1
#k1_2
#data_centroid1.shape


# In[53]:


# Get initial Centroids for k2
import pandas as pd
# get 2nd centroid
data_centroid1 = pd.DataFrame(data, columns = ['x', 'y'])
data_centroid1['distance_from_1'] = numpy.sqrt((data_centroid1['x'] - i_point2[0])**2 + (data_centroid1['y'] - i_point2[1])**2)
k1_2 = numpy.argmax(data_centroid1['distance_from_1'])
k1_2_list = [data_centroid1['x'][k1_2], data_centroid1['y'][k1_2]]
i_point1_list = numpy.zeros((6, 2))
i_point1_list[0] = i_point2
i_point1_list[1] = k1_2_list
data_centroid1 = data_centroid1.drop([k1_2])

# get 3rd centroid
data_centroid1['distance_from_2'] = numpy.sqrt((data_centroid1['x'] - k1_2_list[0])**2 + (data_centroid1['y'] - k1_2_list[1])**2)
data_centroid1['distance_from_12'] = (data_centroid1['distance_from_2'] 
                                      + data_centroid1['distance_from_1']) / 2
k1_3 = numpy.argmax(data_centroid1['distance_from_12'])
k1_3_list = [data_centroid1['x'][k1_3], data_centroid1['y'][k1_3]]
i_point1_list[2] = k1_3_list
data_centroid1 = data_centroid1.drop([k1_3])
                                     

# get 4th centroid
data_centroid1['distance_from_3'] = numpy.sqrt((data_centroid1['x'] - k1_3_list[0])**2 + (data_centroid1['y'] - k1_3_list[1])**2)
data_centroid1['distance_from_123'] = (data_centroid1['distance_from_2'] + data_centroid1['distance_from_1'] 
                                       + data_centroid1['distance_from_3']) / 3
k1_4 = numpy.argmax(data_centroid1['distance_from_123'])
k1_4_list = [data_centroid1['x'][k1_4], data_centroid1['y'][k1_4]]
i_point1_list[3] = k1_4_list
data_centroid1 = data_centroid1.drop([k1_4])
 
    
# get 5th centroid
data_centroid1['distance_from_4'] = numpy.sqrt((data_centroid1['x'] - k1_4_list[0])**2 + (data_centroid1['y'] - k1_4_list[1])**2)
data_centroid1['distance_from_1234'] = (data_centroid1['distance_from_2'] + data_centroid1['distance_from_1'] 
                                        + data_centroid1['distance_from_3'] + data_centroid1['distance_from_4']) / 4
k1_5 = numpy.argmax(data_centroid1['distance_from_1234'])
k1_5_list = [data_centroid1['x'][k1_5], data_centroid1['y'][k1_5]]
i_point1_list[4] = k1_5_list
data_centroid1 = data_centroid1.drop([k1_5])


# get 6th centroid
data_centroid1['distance_from_5'] = numpy.sqrt((data_centroid1['x'] - k1_5_list[0])**2 + (data_centroid1['y'] - k1_5_list[1])**2)
data_centroid1['distance_from_12345'] = (data_centroid1['distance_from_2'] + data_centroid1['distance_from_1'] 
                                        + data_centroid1['distance_from_3'] + data_centroid1['distance_from_4'] 
                                         + data_centroid1['distance_from_5']) / 5
k1_6 = numpy.argmax(data_centroid1['distance_from_12345'])
k1_6_list = [data_centroid1['x'][k1_6], data_centroid1['y'][k1_6]]
i_point1_list[5] = k1_6_list


    
i_point2 = i_point1_list
i_point2
#k1_2
#data_centroid1.shape


# Kmean for K1, K = 4

# In[54]:


#Visualize initialization centroid on data
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(data[:,0], data[:,1], s=40, color='k')
colmap = {1:'r', 2:'g', 3:'b', 4:'y', 5:'m', 6:'c'}
for i in range(len(i_point1)):
    plt.scatter(i_point1[i,0], i_point1[i,1], color=colmap[i + 1])
plt.show()


# In[55]:


# Assignment 
import pandas as pd
ipoints = {}
i_point1_list = i_point1.tolist()
for i in range(len(i_point1_list)):
    ipoints.setdefault(i+1,[]).append(i_point1_list[i])
    
data = pd.DataFrame(data, columns = ['x', 'y'])
def dist_toCentroid(df, centroids):
    for i in range(len(centroids)):
        df['distance_from_{}'.format(i+1)] = numpy.sqrt((df['x'] - centroids[i+1][0][0])**2 + (df['y'] - centroids[i+1][0][1])**2)
    centroid_dist_cols = ['distance_from_{}'.format(i+1) for i in range(len(centroids))]
    df['closest'] = df.loc[:, centroid_dist_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    cluster1 = df[df['closest']==1][['x','y']]
    cl1 = (numpy.mean(cluster1, axis=0)).tolist()
    cluster2 = df[df['closest']==2][['x','y']]
    cl2 = (numpy.mean(cluster2, axis=0)).tolist()
    cluster3 = df[df['closest']==3][['x','y']]
    cl3 = (numpy.mean(cluster3, axis=0)).tolist()
    newCentroid_l = [cl1, cl2, cl3]
    newCentroids = {}
    if len(centroids) > 3:
        cluster4 = df[df['closest']==4][['x','y']]
        cl4 = (numpy.mean(cluster4, axis=0)).tolist()
        newCentroid_l.append(cl4)
    if len(centroids) > 4:
        cluster5 = df[df['closest']==5][['x','y']]
        cl5 = (numpy.mean(cluster5, axis=0)).tolist()
        newCentroid_l.append(cl5)
    if len(centroids) > 5:
        cluster6 = df[df['closest']==6][['x','y']]
        cl6 = (numpy.mean(cluster6, axis=0)).tolist()
        newCentroid_l.append(cl6)
    for i in range(len(newCentroid_l)):
        newCentroids.setdefault(i+1,[]).append(newCentroid_l[i])
    df['color'] = df['closest']
    df.loc[df['closest'] == 1, 'color'] = 'r'
    df.loc[df['closest'] == 2, 'color'] = 'g'
    df.loc[df['closest'] == 3, 'color'] = 'b'
    df.loc[df['closest'] == 4, 'color'] = 'y'
    df.loc[df['closest'] == 5, 'color'] = 'm'
    df.loc[df['closest'] == 6, 'color'] = 'c'
    
    return newCentroids, df

print(ipoints)
nC, df1 = dist_toCentroid(data, ipoints)


# In[56]:


print(df1.head())
plt.scatter(df1['x'], df1['y'], s=40, color=df1['color'])
for i in nC.keys():
    plt.scatter(nC[i][0][0], nC[i][0][1], color=colmap[i], edgecolor='k', s=140)
plt.show()
print(nC)


# In[57]:


q1 = 5
while q1 > 0:
    oldClosest = df1['closest']
    ipointA = nC
    nC, df1 = dist_toCentroid(data, ipointA)
    df1['oldClosest'] = oldClosest
    df1['q1'] = df1['oldClosest'] - df1['closest']
    q1 = numpy.sum(numpy.absolute(df1['q1']))
    
print(nC)


# In[58]:


# Results - mean vectors and objective function
meanVectors = nC
cluster1 = df1[df1['closest']==1][['x','y']]
cluster1_squared_error = numpy.sum((cluster1['x'] - meanVectors[1][0][0])**2 + (cluster1['y'] - meanVectors[1][0][1])**2)

cluster2 = df1[df1['closest']==2][['x','y']]
cluster2_squared_error = numpy.sum((cluster2['x'] - meanVectors[2][0][0])**2 + (cluster2['y'] - meanVectors[2][0][1])**2)

cluster3 = df1[df1['closest']==3][['x','y']]
cluster3_squared_error = numpy.sum((cluster3['x'] - meanVectors[3][0][0])**2 + (cluster3['y'] - meanVectors[3][0][1])**2)

cluster4 = df1[df1['closest']==4][['x','y']]
cluster4_squared_error = numpy.sum((cluster4['x'] - meanVectors[4][0][0])**2 + (cluster4['y'] - meanVectors[4][0][1])**2)

sum_squared_error = cluster1_squared_error + cluster2_squared_error + cluster3_squared_error + cluster4_squared_error

print(meanVectors)
print(sum_squared_error)


plt.scatter(df1['x'], df1['y'], s=40, color=df1['color'])
for i in nC.keys():
    plt.scatter(nC[i][0][0], nC[i][0][1], color=colmap[i], edgecolor='k', s=140)
plt.show()
df1


# Kmean for K2, K = 6

# In[59]:


#Visualize initialization centroid on data
plt.scatter(data['x'], data['y'], s=40, color='k')
for i in range(len(i_point2)):
    plt.scatter(i_point2[i,0], i_point2[i,1], color=colmap[i + 1])
plt.show()


# In[60]:


ipoints2 = {}
i_point2_list = i_point2.tolist()
for i in range(len(i_point2_list)):
    ipoints2.setdefault(i+1,[]).append(i_point2_list[i])
    
print(ipoints2)
nC, df2 = dist_toCentroid(data, ipoints2)
nC


# In[61]:


print(df2.head())
plt.scatter(df2['x'], df2['y'], s=40, color=df2['color'])
for i in nC.keys():
    plt.scatter(nC[i][0][0], nC[i][0][1], color=colmap[i], edgecolor='k', s=140)
plt.show()
print(nC)


# In[62]:


q1 = 5
while q1 > 0:
    oldClosest = df2['closest']
    ipointA = nC
    nC, df2 = dist_toCentroid(data, ipointA)
    df2['oldClosest'] = oldClosest
    df2['q1'] = df2['oldClosest'] - df2['closest']
    q1 = numpy.sum(numpy.absolute(df2['q1']))
    
print(nC)


# In[63]:


# Results - mean vectors and objective function
meanVectors = nC
cluster1 = df2[df2['closest']==1][['x','y']]
cluster1_squared_error = numpy.sum((cluster1['x'] - meanVectors[1][0][0])**2 + (cluster1['y'] - meanVectors[1][0][1])**2)

cluster2 = df2[df2['closest']==2][['x','y']]
cluster2_squared_error = numpy.sum((cluster2['x'] - meanVectors[2][0][0])**2 + (cluster2['y'] - meanVectors[2][0][1])**2)

cluster3 = df2[df2['closest']==3][['x','y']]
cluster3_squared_error = numpy.sum((cluster3['x'] - meanVectors[3][0][0])**2 + (cluster3['y'] - meanVectors[3][0][1])**2)

cluster4 = df2[df2['closest']==4][['x','y']]
cluster4_squared_error = numpy.sum((cluster4['x'] - meanVectors[4][0][0])**2 + (cluster4['y'] - meanVectors[4][0][1])**2)

cluster5 = df2[df2['closest']==5][['x','y']]
cluster5_squared_error = numpy.sum((cluster5['x'] - meanVectors[5][0][0])**2 + (cluster5['y'] - meanVectors[5][0][1])**2)


cluster6 = df2[df2['closest']==6][['x','y']]
cluster6_squared_error = numpy.sum((cluster6['x'] - meanVectors[6][0][0])**2 + (cluster6['y'] - meanVectors[6][0][1])**2)



sum_squared_error = cluster1_squared_error + cluster2_squared_error + cluster3_squared_error + cluster4_squared_error + cluster5_squared_error + cluster6_squared_error

print(meanVectors)
print(sum_squared_error)


plt.scatter(df2['x'], df2['y'], s=40, color=df1['color'])
for i in nC.keys():
    plt.scatter(nC[i][0][0], nC[i][0][1], color=colmap[i], edgecolor='k', s=140)
plt.show()
df2


# Cross check with sklearn

# In[43]:


data2 = data[['x', 'y']]
data2.shape


# In[44]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init=i_point1, n_init=1).fit(data2)
kmeans.cluster_centers_


# In[45]:


kmeans.inertia_


# In[46]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, init=i_point2, n_init=1).fit(data2)
kmeans.cluster_centers_


# In[47]:


kmeans.inertia_

