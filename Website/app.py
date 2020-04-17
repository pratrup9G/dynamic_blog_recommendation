##Import Libraries
import numpy as np
import pandas as pa
import pickle
from flask import Flask,render_template,request
from selenium import webdriver
import demoji #(pip install demoji) after that demoji.download_codes()
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize,TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import os

#import nltk
from scipy.stats import entropy
demoji.download_codes() # (Required for removing emojis from a text data)

app = Flask(__name__)


#GOOGLE_CHROME_PATH = '/app/.apt/usr/bin/google_chrome'
#CHROMEDRIVER_PATH = '/app/.chromedriver/bin/chromedriver'


#Load the trained models using pickle
lda = pickle.load(open('lda_model','rb'))
dictionary = pickle.load(open('dictonary','rb'))
corpus = pickle.load(open('corpus','rb'))

## Processing Text
train_data = pa.read_csv('training_data.csv')
train_data.reset_index(drop=True,inplace=True)


## This function will convert all the numbers like 1000 to thousand
ones = ["", "one ","two ","three ","four ", "five ", "six ","seven ","eight ","nine ","ten ","eleven ",
        "twelve ", "thirteen ", "fourteen ", "fifteen ","sixteen ","seventeen ", "eighteen ","nineteen "]
twenties = ["","","twenty ","thirty ","forty ", "fifty ","sixty ","seventy ","eighty ","ninety "]
thousands = ["","thousand ","million ", "billion ", "trillion ", "quadrillion ", "quintillion ", "sextillion ",
             "septillion ","octillion ", "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ",
             "quattuordecillion ", "quindecillion", "sexdecillion ", "septendecillion ", "octodecillion ", "novemdecillion ",
             "vigintillion "]
def num999(n):
    c = int(n % 10) # singles digit
    b = int(((n % 100) - c) / 10) # tens digit
    a = int(((n % 1000) - (b * 10) - c) / 100) # hundreds digit
    t = ""
    h = ""
    if a != 0 and b == 0 and c == 0:
        t = ones[a] + "hundred "
    elif a != 0:
        t = ones[a] + "hundred and "
    if b <= 1:
        h = ones[n%100]
    elif b > 1:
        h = twenties[b] + ones[c]
    st = t + h
    return st
def num2word(num):
    if num == 0: return 'zero'
    i = 3
    n = str(num)
    word = ""
    k = 0
    while(i == 3):
        nw = n[-i:]
        n = n[:-i]
        if int(nw) == 0:
            word = num999(int(nw)) + thousands[int(nw)] + word
        else:
            word = num999(int(nw)) + thousands[k] + word
        if n == '':
            i = i+1
        k += 1
    return word[:-1]
def subs(word):
    number = word.group(0)
    return (num2word(number))
def find_numbers_percent(word):
    return re.sub('[\d]+',subs,word)


## Convert all Apostrophe 
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"we'll" : "we will"   ,
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

## Tokenizers 
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()
engstopwords = set(stopwords.words('english'))
punc = string.punctuation  
## emojis wriiten using brakets remove this emojis
emo = {'):','(:',':',':-)',':))'}
    
## Cleaning the data
def data_cleaning(data):
    #Remove email
    re_email = re.compile(r'[\w.-]+@[\w.-]+')
    data= re_email.sub(r'',data)
    #Remove Emoji
    data = demoji.replace(data,repl='')
    #Remove webiste
    reg_website = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([\w+-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    data= reg_website.sub(r'',data)
    #convert lower case
    data = data.lower()
    #convert numbers to words
    data = find_numbers_percent(data)
    #convert percent
    data = re.sub(r'%','percent',data)
    data = re.sub(r'’',"'",data)
    data = re.sub(r'\.',"",data)
    data = re.sub(r'-',"",data)
    data = re.sub(r'–',"",data)
    data = re.sub(r'”',"",data)
    data = re.sub(r':',"",data)
    data = re.sub(r'‘',"",data)
    data = re.sub(r'“',"",data)
    # tokenize words
    words = tokenizer.tokenize(data)
    # add appos
    words = [APPO[word] if word in APPO else word for word in words]
    # stop words
    words = [word for word in words if not word in engstopwords]
    words = [lem.lemmatize(word,'v') for word in words]
    #Remove Punctuanitions
    #clean_data = [str('') if word == '.' else word for word in words]
    clean_data = [word for word in words if word not in punc]
    #clean_data = [word for word in words if wor]
    clean_data = [word for word in clean_data if word not in emo]
    cleaned_data =  " ".join(clean_data)
    
    #cleaned_data = re.sub(r'.','dot',cleaned_data)
    cleaned_data = re.sub(r"'",'',cleaned_data)
    return cleaned_data


##Web scraping using selinium (When the user will paste the link of the blog all the data in that blog will be scraped using this function)
def web_scrapper(link):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)

    driver.get(link)
    description_p = driver.find_elements_by_tag_name('p')
    doc = []
    for para in description_p:
        doc.append(para.text)
    
    return "".join(doc)


## This function will calcuate the distance of  topic distributions of a document with other documents (Here documents are blogs) 
def jensen_shannon(query, matrix):
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

## Find the most similar documents 
def get_most_similar_documents(query,matrix,k=100):
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances


## Routing function
@app.route('/')
def home():
    return render_template('dynamic_blog.html')

@app.route('/recommendation',methods=["GET","POST"])
def recommendation():
    #Get all the request from form
    link = request.form['url']
    
    
    #Scrap that website I used selinium (Beautiful soup can also be used)
    scraped_text = web_scrapper(link)
    
    ## Radio buttons inputs
    medium_checked = request.form.get("medium_check") != None
    analytics_checked = request.form.get("analytics_check") != None
    towards_checked = request.form.get("towards_check") != None

    
    ##Clean the data
    cleaned_test_data = data_cleaning(scraped_text)
    cleaned_test_data = tokenizer.tokenize(cleaned_test_data)
    
    
    test_bow = dictionary.doc2bow(cleaned_test_data)
    
    ## Get the topic distribution of test data
    test_topic_dist = np.array([tup[1]  for tup in lda.get_document_topics(bow=test_bow)]) 
    ## All the topic distribution of train data
    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    ## Find the most similar documents
    most_sim_ids = get_most_similar_documents(test_topic_dist,doc_topic_dist)
    most_similar_documents = train_data[train_data.index.isin(most_sim_ids)]
    
    
    titles = []
    links = []
    webpage = []
    
    
    ## Take only the similar documents based on the users input whether from analytics vidhya or from medium or from towards data science
    
    if medium_checked and not analytics_checked and not towards_checked:
        most_similar_documents = most_similar_documents[most_similar_documents['Webpage'] == 'Medium']

    elif analytics_checked and not towards_checked and not medium_checked:
        most_similar_documents =most_similar_documents[most_similar_documents['Webpage'] == 'Analytics_Vidhya']
        #print(most_similar_documents)
        
    elif towards_checked and not medium_checked and not analytics_checked:
        most_similar_documents =most_similar_documents[most_similar_documents['Webpage'] == 'Towards_Data_Science']
        
    elif (medium_checked and analytics_checked and not towards_checked):
        most_similar_documents =most_similar_documents[(most_similar_documents['Webpage'] == 'Analytics_Vidhya') | (most_similar_documents['Webpage'] == 'Medium')]
        #print(most_similar_documents)
        
    elif (analytics_checked and towards_checked and not medium_checked):
        most_similar_documents =most_similar_documents[(most_similar_documents['Webpage'] == 'Analytics_Vidhya') | (most_similar_documents['Webpage'] == 'Towards_Data_Science')]
    
    elif (medium_checked and towards_checked and not analytics_checked):
        
        most_similar_documents.to_csv('most_sim.csv')
        most_similar_documents =most_similar_documents[(most_similar_documents['Webpage'] == 'Medium') | (most_similar_documents['Webpage'] == 'Towards_Data_Science')]
    
    elif (medium_checked and towards_checked and analytics_checked):
        most_similar_documents =most_similar_documents[(most_similar_documents['Webpage'] == 'Medium') | (most_similar_documents['Webpage'] == 'Towards_Data_Science') | (most_similar_documents['Webpage'] == 'Analytics_Vidhya')]
    
    if len(most_similar_documents) > 9:
    #Append the titles and links and webpage of the most_similar_documents in titles and links and send it to webpage
        for i in range(0,10):
                titles.append(most_similar_documents['Title'].iloc[i])
                links.append(most_similar_documents['Links'].iloc[i])
                webpage.append(most_similar_documents['Webpage'].iloc[i])
        # print('******************************')
    else:
        for i in range(0,len(most_similar_documents)):
                titles.append(most_similar_documents['Title'].iloc[i])
                links.append(most_similar_documents['Links'].iloc[i])
                webpage.append(most_similar_documents['Webpage'].iloc[i])
                
    return render_template('dynamic_blog.html',titles_links=zip(titles,links,webpage))

if __name__ == "__main__":
    app.run()
    
