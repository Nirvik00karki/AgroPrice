import json
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.http import JsonResponse
from django.shortcuts import render

def generate_actual_vs_predicted_plot(actual_prices, predicted_prices):
    actual_prices = np.exp(actual_prices)
    predicted_prices = np.exp(predicted_prices)

    plt.plot(actual_prices, label='Actual Prices', marker='o')
    plt.plot(predicted_prices, label='Predicted Prices', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()

    # Convert plot to base64 for embedding in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    return plot_data

def analysis_fig(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_commodity = data.get('commodity')
        if not selected_commodity:
            return JsonResponse({'error': 'No commodity selected for analysis'})

        # Get actual and predicted prices from the context
        actual_prices = data.get('actual_prices', [])
        predicted_prices = data.get('predicted_prices', [])

        # Generate actual vs predicted plot
        actual_vs_predicted_plot = generate_actual_vs_predicted_plot(actual_prices, predicted_prices)

        # You can add more visualization functions here based on your requirements

        # Render the analysis template with the generated plots
        context = {
            'commodity': selected_commodity,
            'actual_vs_predicted_plot': actual_vs_predicted_plot,
            # Add more plot data to the context as needed
        }

        return render(request, 'analysis.html', context)
    else:
        return JsonResponse({'error': 'Invalid request method'})

def predict_average_price(request):
    print("Predict average price function called")
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_date = data.get('date')
        print(f'Commodity date: {selected_date}')
        selected_commodity = data.get('commodity')
        print(f'Commodity name: {selected_commodity}')
        if not selected_commodity:
            return JsonResponse({'error': 'No commodity selected'})

        df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\lagadded_out.csv')

        # Filter data for the selected commodity
        df1 = df[df['Commodity'] == selected_commodity]

        # Check if data is available for the selected commodity
        if df1.empty:
            return JsonResponse({'error': f'No data found for {selected_commodity}'})
     
        df1['Date'] = pd.to_datetime(df1['Date'])

        df1 = df1.sort_values('Date')
        df1 = df1[(df1['Date'] < '2020-01-01') | (df1['Date'] >= '2021-01-01')]
        latest_months = 6  # Set the number of latest months for the test data

        latest_date = df1['Date'].max()
        earliest_date = latest_date - pd.DateOffset(months=latest_months)

# Create training and test datasets
        train_data = df1[df1['Date'] < earliest_date]
        test_data = df1[df1['Date'] >= earliest_date]

# Separate features and target variables for training and test datasets
        x_train = train_data.drop(['Average', 'Commodity', 'Date'], axis=1)  # Assuming 'Average' is the target variable
        y_train = train_data['Average']
        x_test = test_data.drop(['Average', 'Commodity', 'Date'], axis=1)
        y_test = test_data['Average']
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Prepare for Ridge Regression (add bias term)
        X_train = np.c_[np.ones(x_train_scaled.shape[0]), x_train_scaled]
        X_test = np.c_[np.ones(x_test_scaled.shape[0]), x_test_scaled]
        print("Shapes before training - X_train:", X_train.shape, "X_test:", X_test.shape)

        # Train the Ridge Regression model
        coefficients = ridge_regression_fit(X_train, y_train, alpha=1.0)
        print("Shapes before prediction - X_test:", X_test.shape)

        # Make predictions
        y_pred = ridge_regression_predict(coefficients, X_test)     
        # ridge_model = Ridge(alpha=1.0)
        # ridge_model.fit(x_train, y_train)

        # lmodel = LinearRegression()
        # lmodel.fit(x_train, y_train)

        # model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
        # model.fit(x_train, y_train)
        # y_pred = ridge_model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        try:
            selected_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
        except ValueError:
            return JsonResponse({'error': 'Invalid date format'})
        
        selected_commodity_lag = df1['Average'].iloc[-1]
        selected_date_features = pd.DataFrame({
            'day': [int(selected_date.day)],
            'month': [int(selected_date.month)],
            'year': [int(selected_date.year)],
            'Season_Fall': [0], 
            'Season_Spring': [0],
            'Season_Summer': [0],
            'Season_Winter': [1],
            'Apple_Jholey': [0],
            'Banana': [0],
            'Carrot_Local': [0],
            'Cucumber_Local': [0],
            'Garlic_Dry_Nepali': [0],
            'Lettuce': [0],
            'Onion_Dry_Indian': [0],
            'Potato_White': [0],
            'Tomato_Big_Nepali': [0],
            'Festival_Buddha_Jayanti': [0],
            'Festival_Dashain': [0],
            'Festival_Gai_Jatra': [0],
            'Festival_Ghode_Jatra': [0],
            'Festival_Holi': [0],
            'Festival_Indra_Jatra': [0],
            'Festival_Janai_Purnima': [0],
            'Festival_Lhosar': [0],
            'Festival_Maghe_Sankranti': [0],
            'Festival_Maha_Shivaratri': [0],
            'Festival_Shree_Panchami': [0],
            'Festival_Teej': [0],
            'Festival_Tihar': [0],
            'Festival_nan': [0],
            'Dashain_near': [0],
            'Tihar_near': [0],
            'Holi_near': [0],
            'Maha_Shivaratri_near': [0],
            'Buddha_Jayanti_near': [0],
            'Ghode_Jatra_near': [0],
            'Teej_near': [0],
            'Indra_Jatra_near': [0],
            'Lhosar_near': [0],
            'Janai_Purnima_near': [0],
            'Gai_Jatra_near': [0],
            'Maghe_Sankranti_near': [0],
            'Shree_Panchami_near': [0],
            'Fall_near': [0],
            'Spring_near': [0],
            'Summer_near': [0],
            'Winter_near': [0],
            'Average_lag1' : [selected_commodity_lag],
            # 'Weighted_Moving_Average' : [0],
})

        selected_date_features = selected_date_features.values.reshape(1, -1)
        print("Selected Date Features:", selected_date_features)
        selected_date_features_scaled = scaler.transform(selected_date_features)
        selected_date_features_scaled_with_bias = np.c_[np.ones(selected_date_features_scaled.shape[0]), selected_date_features_scaled]
        # selected_date_features_with_bias = np.c_[np.ones(selected_date_features.shape[0]), selected_date_features]
        print("Selected Date Features with Bias:", selected_date_features_scaled_with_bias)    
        prediction = ridge_regression_predict(coefficients, selected_date_features_scaled_with_bias)
        # prediction = ridge_model.predict(selected_date_features)
        print("Raw Prediction:", prediction)
        predicted_average_price = np.exp(prediction[0])
        print("Predicted Price:", predicted_average_price)

        # actual_data = get_commodity_data(request, json.dumps([selected_commodity]))
        # actual_prices = actual_data['commodity_data'][selected_commodity]['average_price']

        # Return the predicted and actual prices along with other data
        context = {
            'predicted_price': predicted_average_price,
            'predicted_prices': json.dumps(list(np.exp(y_pred))),
            # 'actual_prices': json.dumps(actual_prices),
            'mse': mse,
            'r2': r2,
            'date': selected_date,
            'commodity': selected_commodity,
        }
    # Render the result.html template with the provided context
        return render(request, 'result.html', context)
    else:
        return JsonResponse({'error': 'Invalid request method'})
