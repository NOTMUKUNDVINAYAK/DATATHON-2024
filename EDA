import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from sklearn.cluster import KMeans

df = pd.read_csv('/content/Train.csv')

print(df.describe())
print(df.info())

mean_length_by_category = df.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean().sort_values(ascending=False)
print(mean_length_by_category)

count_vectorizer = CountVectorizer(max_features=100)  # Limit to 100 features
count_matrix = count_vectorizer.fit_transform(df['ENTITY_DESCRIPTION'])
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
print(count_df.head())  # Display the first few rows of the count DataFrame

text = ' '.join(df['ENTITY_DESCRIPTION'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for ENTITY_DESCRIPTION')
plt.show()

feature1 = count_df.columns[0]  
feature2 = count_df.columns[1]  

plt.figure(figsize=(10, 6))
plt.scatter(count_df[feature1], count_df[feature2], c=df['ENTITY_LENGTH'], cmap='viridis', alpha=0.6)
plt.colorbar(label='ENTITY_LENGTH')
plt.title('Scatter Plot of ENTITY_LENGTH vs Count Vectorized Features')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.show()

total_categories = df['CATEGORY_ID'].nunique()
print(f'Total unique categories: {total_categories}')

numeric_df = df.select_dtypes(include=[np.number])  
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

vectorizer = TfidfVectorizer(ngram_range=(2, 3)) 
ngram_matrix = vectorizer.fit_transform(df['ENTITY_DESCRIPTION'])
print(f'TF-IDF Matrix Shape: {ngram_matrix.shape}')

kmeans = KMeans(n_clusters=5, random_state=42) 
clusters = kmeans.fit_predict(ngram_matrix)
df['Cluster'] = clusters  # Add cluster labels to the DataFrame
print(df[['ENTITY_ID', 'CATEGORY_ID', 'Cluster']].head())  
