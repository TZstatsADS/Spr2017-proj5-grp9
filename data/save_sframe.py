import graphlab as gl

# this script generates a master sframe 'msf' which we can use for our analysis

r = gl.SFrame.read_csv('reviews_Kindle_Store.json', delimiter='\n', header=False)

reviews = r.unpack(unpack_column='X1',column_name_prefix='')
reviews = reviews.unpack(unpack_column='helpful',column_name_prefix='X')
reviews.rename({'X.0':'upvotes','X.1':'downvotes'})
reviews['reviewTime'] = reviews['reviewTime'].str_to_datetime(str_format="%m %d, %Y")
reviews['tfidf'] = gl.text_analytics.tf_idf(reviews['reviewText'])
reviews['tfidf'] = reviews['tfidf'].dict_trim_by_keys(gl.text_analytics.stopwords(), True)

d = gl.SFrame.read_csv('meta_Kindle_Store.json', delimiter='\n', header=False)

meta = d.unpack(unpack_column='X1',column_name_prefix='')
msf = reviews.join(meta, on='asin', how='inner')

msf.save('kindle_data.sf')
