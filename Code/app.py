from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: ("Rice", "rice.jpg"), 2: ("Maize", "maize.jpg"), 3: ("Jute", "jute.jpg"), 
    4: ("Cotton", "cotton.jpg"), 5: ("Coconut", "coconut.jpg"), 6: ("Papaya", "papaya.jpg"), 
    7: ("Orange", "orange.jpg"), 8: ("Apple", "apple.jpg"), 9: ("Muskmelon", "muskmelon.jpg"), 
    10: ("Watermelon", "watermelon.jpg"), 11: ("Grapes", "grapes.jpg"), 12: ("Mango", "mango.jpg"), 
    13: ("Banana", "banana.jpg"), 14: ("Pomegranate", "pomegranate.jpg"), 15: ("Lentil", "lentil.jpg"), 
    16: ("Blackgram", "blackgram.jpg"), 17: ("Mungbean", "mungbean.jpg"), 18: ("Mothbeans", "mothbeans.jpg"), 
    19: ("Pigeonpeas", "pigeonpeas.jpg"), 20: ("Kidneybeans", "kidneybeans.jpg"), 
    21: ("Chickpea", "chickpea.jpg"), 22: ("Coffee", "coffee.jpg"), 23: ("Sugarcane", "sugarcane.jpg"),
    24: ("Guava", "guava.jpg"), 25: ("Ragi", "ragi.jpg")
    }

    if prediction[0] in crop_dict:
        crop, image_file = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
        image_url = f"static/{image_file}"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result, image_url = image_url)




# python main
if __name__ == "__main__":
    app.run(debug=True)