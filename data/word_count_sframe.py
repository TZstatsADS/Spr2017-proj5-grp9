import graphlab as gl

msf = gl.load_sframe('kindle_data.sf/')

# concatenate the different fields into a single text field
docs = msf.apply(lambda x: str(x['summary']) + ' ' + str(x['reviewText']) + ' ' + str(x['description']))
print("concatenation finished, starting tokenization")

# tokenize the text
if not docs.is_materialized():
    print("materializing the docs")
    docs.__materialize__()
    print("finished materializing docs")
print("starting tokenizing")
text = gl.text_analytics.tokenize(docs, to_lower=True)
print("tokenizing finished, starting text transformation")

# get english stop words into a list
stp = list(gl.text_analytics.stopwords(lang='en'))

# transform the text by removing rare words and stop words
sf = gl.SFrame({'text' : text})
wt = gl.toolkits.feature_engineering.RareWordTrimmer('text',
                                                     threshold=1,
                                                     stopwords=stp)
print("starting fit.transformer")
fit_wt = wt.fit(sf) # fit transformer
transformed_sf = fit_wt.transform(sf)

# now we count the words
print("finished fit transform, materializing transformed_sf")
transformed_sf['word_count'] = gl.text_analytics.count_words(transformed_sf['text'])

# save the resulting SFrame for use by topic model
print("saving sframe to disk")
transformed_sf.save('word_counts.sf')