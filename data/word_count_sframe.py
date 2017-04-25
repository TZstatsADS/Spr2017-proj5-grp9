import graphlab as gl

msf = gl.load_sframe('kindle_data.sf/')

# concatenate the different fields into a single text field
docs = msf.apply(lambda x: str(x['summary']) + ' ' + str(x['reviewText']) + ' ' + str(x['description']))

# tokenize the text
text = gl.text_analytics.tokenize(docs, to_lower=True)

# get english stop words into a list
stp = list(gl.text_analytics.stopwords(lang='en'))

# transform the text by removing rare words and stop words
sf = gl.SFrame({'text' : text})
wt = gl.toolkits.feature_engineering.RareWordTrimmer('text',
                                                     threshold=1,
                                                     stopwords=stp)
fit_wt = wt.fit(sf) # fit transformer
transformed_sf = fit_wt.transform(sf)

# now we count the words
transformed_sf['word_count'] = gl.text_analytics.count_words(transformed_sf['text'])

# save the resulting SFrame for use by topic model
transformed_sf.save('word_counts.sf')