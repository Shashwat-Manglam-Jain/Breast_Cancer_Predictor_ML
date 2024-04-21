from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model using pickle
model = pickle.load(open('BreastModel.pkl', 'rb'))

# Create the Flask application
app = Flask(__name__)

# Define fallback image URL
fallback_image_url = "https://t3.ftcdn.net/jpg/03/18/72/22/240_F_318722204_0XEZRKrwT2EidInGEB00VYQHSSdEC7m2.jpg"

# Index route
@app.route('/')
def index():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Define the feature names in the order the model expects them
    feature_names = [
         "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    try:
        # Initialize a list to store the feature values
        features = []
        
        # Extract features from the form and convert them to floats
        for name in feature_names:
            value = request.form.get(name)
            
            # Check if the form data is missing or invalid
            if value is None or value.strip() == '':
                raise ValueError(f'Missing or invalid input for feature: {name}')
            
            # Convert the form value to a float and add to the features list
            features.append(float(value))
        
        # Convert the list of features to a numpy array and reshape it for the model
        feature_np = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(feature_np)[0]
        
        # Determine the result and image path based on the prediction
        if prediction == 1:
            message = "Cancerous"
            image_url = "https://sunnybrook.ca/uploads/1/_research/news/2017/breast-cancer-female_48876817_for-web.jpg"  # URL of the cancerous image
            message_text = "You are infected. Please consult a doctor as soon as possible."
        else:
            message = "Not Cancerous"
            image_url = "https://cdn-assets-eu.frontify.com/s3/frontify-enterprise-files-eu/eyJvYXV0aCI6eyJjbGllbnRfaWQiOiJmcm9udGlmeS1maW5kZXIifSwicGF0aCI6ImloaC1oZWFsdGhjYXJlLWJlcmhhZFwvYWNjb3VudHNcL2MzXC80MDAwNjI0XC9wcm9qZWN0c1wvMjA5XC9hc3NldHNcLzc3XC8zNzM5OVwvNDhlMmQ4MmY2ZDNmNmUzODhhMDczZmNlMjQ3NDc4OGUtMTY1ODI5ODk1Ny5qcGcifQ:ihh-healthcare-berhad:rB8OCeWXkmC9-RQxTMGVNFpD19k_dSx1alwBQW93ejk?format=webp"  # URL of the non-cancerous image
            message_text = "You are fit and fine."
        
        # Render the index.html template with the prediction result
        return render_template('index.html', message=message, image_url=image_url, message_text=message_text)
    
    except ValueError as e:
        # Handle ValueError and return a fallback image URL and error message
        return render_template('index.html',  message=f'Error: {str(e)}', image_url=fallback_image_url, message_text="An error occurred. Please review your inputs and try again.")
    except Exception as e:
        # Handle any other exceptions and return a fallback image URL and error message
        return render_template('index.html', message=f'Error: {str(e)}', image_url=fallback_image_url, message_text="An unexpected error occurred. Please try again later.")

# Main function to run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
