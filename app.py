from flask import Flask, render_template, request, redirect, url_for
from src.DelhiHousingProject.exception import CustomException
from src.DelhiHousingProject.pipelines.prediction_pipeline import CustomData,PredictPipeline
import os
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and validate form data
            data = CustomData(
                area=float(request.form.get('area', '0')),
                latitude=float(request.form.get('latitude', '0')),
                longitude=float(request.form.get('longitude', '0')),
                Bedrooms=int(request.form.get('Bedrooms', '0')),
                Bathrooms=int(request.form.get('Bathrooms', '0')),
                Balcony=int(request.form.get('Balcony', '0')),
                neworold=request.form.get('neworold', 'Unknown'),
                parking=int(request.form.get('parking', '0')),
                Furnished_status=request.form.get('Furnished_status', 'Unknown'),
                Lift=int(request.form.get('Lift', '0')),
                type_of_building=request.form.get('type_of_building', 'Unknown')
            )
            pred_df = data.get_data_as_data_frame()
            prediction_pipeline = PredictPipeline()
            results = prediction_pipeline.predict(pred_df)

            return redirect(url_for('result', predicted_price=results[0]))
        except ValueError as e:
            return render_template('predict.html', error=str(e))
        except CustomException as e:
            return render_template('predict.html', error="Prediction error occurred.")
        except Exception as e:
            return render_template('predict.html', error="An unexpected error occurred.")
    else:
        return render_template('predict.html')


@app.route('/result')
def result():
    predicted_price = request.args.get('predicted_price')
    return render_template('result.html', predicted_price=predicted_price)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/author')
def author():
    return render_template('author.html')

@app.route('/help')
def help1():
    return render_template('help.html')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
