from flask import Flask,render_template,request
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
import flair
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from sklearn.cluster import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming
from scipy.spatial.distance import cdist

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process',methods=('POST','GET'))
def process():
    if request.method == 'POST':
        city = request.form['city'] #when using plain text field on index.html
        #city = request.form['value'] #when using drop-down on index.html
        preferences = request.form['preferences']
        num_days = request.form['num_days']
        recommendations = get_recommendations(city,preferences,num_days)
        return render_template('result.html',recommendations=recommendations)


def similarity(d2,tfidf_matrix):
  res =[]
  for i in range(len(tfidf_matrix)):
    dot = np.dot(d2,tfidf_matrix[i])
    b = np.power(tfidf_matrix[i],2)
    c= np.power(d2,2)
    moda = np.sqrt(np.sum(b))
    modb =np.sqrt(np.sum(c))
    result = dot/(moda*modb)
    res.append([i,result])
  return res


def make_query(query,city):
  ans = []

  glove_embedding = WordEmbeddings('glove')

  document_embeddings = DocumentPoolEmbeddings([glove_embedding])
  ##adding spell check parts##
  query = TextBlob(query)
  query = str(query.correct())  
  ##finished adding spell check parts##
  sentence = Sentence(query)
  document_embeddings.embed(sentence)
  matrix1 = sentence.get_embedding().cpu().numpy()
  city = city.lower()
  # print(matrix1)
  if city == 'new delhi':
    delhi_embedding = joblib.load("delhi_embedding.sav")
    delhi_data = pd.read_csv("delhi_data.csv")
    result1 = sorted(similarity(matrix1, delhi_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([delhi_data.iloc[result1[j][0]]['place'], delhi_data.iloc[result1[j][0]]['Latitude'], delhi_data.iloc[result1[j][0]]['Longitude']])#, delhi_data.iloc[result1[j][0]]['city']])
    ans_placesc = [x[0] for x in ans]
    ans_df=delhi_data[delhi_data['place'].isin(ans_placesc)]
    return ans_df
  #elif city == 'chennai':
    #result1 = sorted(similarity(matrix1, chennai_embedding),key=lambda x: x[1],reverse=True)
  elif city == 'kolkata':
    kolkata_embedding = joblib.load("kolkata_embedding.sav")
    kolkata_data = pd.read_csv("kolkata_data.csv")
    result1 = sorted(similarity(matrix1, kolkata_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([kolkata_data.iloc[result1[j][0]]['place'], kolkata_data.iloc[result1[j][0]]['Latitude'], kolkata_data.iloc[result1[j][0]]['Longitude']])#, delhi_data.iloc[result1[j][0]]['city']])
    ans_placesc = [x[0] for x in ans]
    ans_df=kolkata_data[kolkata_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'pune':
    pune_embedding = joblib.load("pune_embedding.sav")
    pune_data = pd.read_csv("pune_data.csv")
    result1 = sorted(similarity(matrix1, pune_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([pune_data.iloc[result1[j][0]]['place'], pune_data.iloc[result1[j][0]]['Latitude'], pune_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=pune_data[pune_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'jaipur':
    jaipur_embedding = joblib.load("jaipur_embedding.sav")
    jaipur_data = pd.read_csv("jaipur_data.csv")
    result1 = sorted(similarity(matrix1, jaipur_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([jaipur_data.iloc[result1[j][0]]['place'], jaipur_data.iloc[result1[j][0]]['Latitude'], jaipur_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=jaipur_data[jaipur_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'udaipur':
    udaipur_embedding = joblib.load("udaipur_embedding.sav")
    udaipur_data = pd.read_csv("udaipur_data.csv")
    result1 = sorted(similarity(matrix1, udaipur_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([udaipur_data.iloc[result1[j][0]]['place'], udaipur_data.iloc[result1[j][0]]['Latitude'], udaipur_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=udaipur_data[udaipur_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'agra':
    agra_embedding = joblib.load("agra_embedding.sav")
    agra_data = pd.read_csv("agra_data.csv")
    result1 = sorted(similarity(matrix1, agra_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([agra_data.iloc[result1[j][0]]['place'], agra_data.iloc[result1[j][0]]['Latitude'], agra_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=agra_data[agra_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'hyderabad':
    hyderabad_embedding = joblib.load("hyderabad_embedding.sav")
    hyderabad_data = pd.read_csv("hyderabad_data.csv")
    result1 = sorted(similarity(matrix1, hyderabad_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([hyderabad_data.iloc[result1[j][0]]['place'], hyderabad_data.iloc[result1[j][0]]['Latitude'], hyderabad_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=hyderabad_data[hyderabad_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'bengaluru':
    bengaluru_embedding = joblib.load("bengaluru_embedding.sav")
    bengaluru_data = pd.read_csv("bengaluru_data.csv")
    result1 = sorted(similarity(matrix1, bengaluru_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([bengaluru_data.iloc[result1[j][0]]['place'], bengaluru_data.iloc[result1[j][0]]['Latitude'], bengaluru_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=bengaluru_data[bengaluru_data['place'].isin(ans_placesc)]
    return ans_df
  elif city == 'mumbai':
    mumbai_embedding = joblib.load("mumbai_embedding.sav")
    mumbai_data = pd.read_csv("mumbai_data.csv")
    result1 = sorted(similarity(matrix1, mumbai_embedding),key=lambda x: x[1],reverse=True)
    for j in range(10):
      ans.append([mumbai_data.iloc[result1[j][0]]['place'], mumbai_data.iloc[result1[j][0]]['Latitude'], mumbai_data.iloc[result1[j][0]]['Longitude']])
    ans_placesc = [x[0] for x in ans]
    ans_df=mumbai_data[mumbai_data['place'].isin(ans_placesc)]
    return ans_df
  else:
    ## this case will never run, as we're providing the user only the above options, to fill from!!
    return None

  


def generate_itinerary(ans_df, num_days):
    obj=KMeans()
    num_days = int(num_days)
    if num_days<=10:
    #handle this exception
        n_c = num_days #number of clusters
        kmeans = KMeans(random_state=0,n_clusters=n_c).fit(ans_df[['Latitude','Longitude']])
    arr=pd.DataFrame(kmeans.cluster_centers_)
    arr.shape

    distance_matrix = cdist(
        arr.values, 
        arr.values,
    )
    distance_matrix

    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    ###permutation 
    # print(permutation)
    # print(distance)

    dict_places = {}
    counter=1
    for i in permutation:
        list_c=np.where(kmeans.labels_==i)
        # print(list_c)
        X_tr=ans_df.iloc[list_c]
        dict_places[counter] = X_tr['place'].to_numpy()
        # print("Travel locations for day",counter)
        # print(X_tr['Place'].to_numpy())
        counter=counter+1
    return dict_places





def get_recommendations(city,preferences,num_days):
    ans_df = make_query(preferences, city)
    place_dict = generate_itinerary(ans_df, num_days)
    return place_dict


if __name__=='__main__':
    app.run(debug=True)