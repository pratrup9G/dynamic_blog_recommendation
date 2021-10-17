# What is the main objective?

The main objective of this project is to recommend blogs to a user based on the similarity of an input blog with other blogs. Suppose I have read a blog about Linear Regression from medium.com and now I want to read blogs of this same topic from analyticsvidhya.com  so in this case just give the link of your blog to this website and it'll automatically recommend you all the blogs from Analytics Vidhya. 

# An introduction to Dynamic Blog recommendation(i.e how it's made and how it works).

* I have used the algorithm **Latent Dirichlet Allocation** for this project. Latent Dirichlet Allocation will find the topic distribution of a particular document(blogs). In simple words suppose a person has written one article about sports so that article will contain different words like football, cricket, basketball so different words belong to different topics. A topic is a collection of words. And we don't know what our topic is, the main aim of this LDA model is to find the topics distribution of a document where topics are a collection of words and a document can contain multiple topics. Words in a particular topic are similar to each other.
(For more read this original research paper ->http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

* For recommendation when the user will give a blog as input first our website will scrap all the details of this blog, clean this blog(removing emojis, removing stopwords, etc) and will give it to the model, then our model will find the topic distribution of this document and will calculate the deviation of this document from other documents(documents present in train data). The lesser the deviation from other documents more is the similarity.

* For training the model I have scraped some blogs from medium.com, Analytics Vidhya, and towards data science. (Total size is 750) 

* I named it dynamic because here I can choose that from which websites I want to get the recommendations.


* LINK WEBSITE https://blog-recommendation-dynamic.herokuapp.com/
