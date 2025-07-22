from flask import Flask, render_template,request,redirect,send_from_directory,url_for
import numpy as np
import json
import uuid
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("models/roomvision_model_updated.h5")
label = ['Bathroom', 'Bedroom', 'Dinning', 'Kitchen', 'Livingroom']

# with open("plant_disease.json",'r') as file:
#     plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image,target_size=(224,224))
    feature = tf.keras.utils.img_to_array(image)
    feature = feature / 255.0
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    class_index = prediction.argmax()
    prediction_label = label[class_index]
    
    return {
        'name': prediction_label,
        'cause': f"The uploaded image is predicted to be a {prediction_label.lower()}.",
        'cure': "Enjoy your beautiful interior!"
    }









@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        print(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}')
        return render_template('home.html',result=True,imagepath = f'/{temp_name}_{image.filename}', prediction = prediction )
    
    else:
        return redirect('/')
        
    
if __name__ == "__main__":
    app.run(debug=True)