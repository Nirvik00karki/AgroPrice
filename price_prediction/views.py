from django.shortcuts import render,redirect
import matplotlib.pyplot as plt
import io
# from io import BytesIO
# from django.conf import settings
import base64
from django.contrib.auth.decorators import login_required
from rest_framework import viewsets
from .models import HistoricalPrice
from .serializers import HistoricalPriceSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.pagination import PageNumberPagination
from django.http import JsonResponse
import pandas as pd
import numpy as np
import os
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import CustomUser
from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

@login_required
def home(request):
    return render(request, 'home.html', {'user': request.user})

def index(request):
    return render(request, 'index.html')


class HistoricalPriceViewSet(viewsets.ModelViewSet):
    queryset = HistoricalPrice.objects.all()
    serializer_class = HistoricalPriceSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = PageNumberPagination

def potato_detail(request):
    return render(request, 'potato.html')

def commodity_detail(request):
    return render(request, 'commodity_detail.html')

def serve_overall_table():
    file_path = os.path.join(r'C:\Final year Project\AgroPrice\price_prediction\static\price_prediction', 'percent.csv')  # Update the path
    overall_table_data = pd.read_csv(file_path).to_dict(orient='records')
    return JsonResponse(overall_table_data, safe=False)

def account(request):
    return render(request, 'account.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')

    return redirect('account') 

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        CustomUser.objects.create_user(username=username, email=email, password=password)

        return redirect('login') 
    else:
        return render(request, 'account.html')


def user_logout(request):
    logout(request)
    return redirect('index') 

def get_commodities(request):
    df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\lagadded_out.csv')
    commodities = df['Commodity'].unique().tolist()
    return JsonResponse(commodities, safe=False)

def get_commodity_data(request, commodity_names):
    commodity_names = json.loads(commodity_names)
    df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\lagadded_out.csv')

    selected_data = df[df['Commodity'].isin(commodity_names)]

    return JsonResponse({
        "commodity_data": {
            commodity: {
                "dates": commodity_data['Date'].astype(str).tolist(),
                "average_price": commodity_data['Average'].tolist(),
            }
            for commodity, commodity_data in selected_data.groupby('Commodity')
        }
    })

def search_commodity(request):
    term = request.GET.get('term')
    df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\lagadded_out.csv')
    search_results = df['Commodity'][df['Commodity'].str.contains(term, case=False)].unique().tolist()
    return JsonResponse(search_results, safe=False)


def ridge_regression_fit(X, y, alpha=1.0):
    # Ridge Regression closed-form solution
    n, m = X.shape
    bias_added = np.allclose(X[:, 0], np.ones(n))

    if not bias_added:
        # If bias term is not added, add it
        X = np.c_[np.ones(n), X]
    identity_matrix = np.identity(m)
    coefficients = np.linalg.inv(X.T @ X + alpha * identity_matrix) @ X.T @ y
    return coefficients

def ridge_regression_predict(coefficients, X):
    predictions = X @ coefficients
    return predictions

# @csrf_exempt
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
        # known_lag_date = df1['Date'].iloc[-1]

# Set known_lag_value to the lag value for selected_commodity on the known_lag_date
        selected_commodity_lag = df1['Average'].iloc[-1]
        known_lag_value = selected_commodity_lag

# Set initial previous_date to one day before the prediction date
        previous_date = selected_date - pd.DateOffset(days=1)

# Initialize previous_date_features with the known lag value
        previous_date_features = pd.DataFrame({
            'day': [int(previous_date.day)],
            'month': [int(previous_date.month)],
            'year': [int(previous_date.year)],
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
            'Average_lag1': [known_lag_value],
        })

    previous_date_features = previous_date_features.values.reshape(1, -1)
    while previous_date.date() < selected_date:
    # Predict lag value for the current previous_date
        previous_date_features_scaled = scaler.transform(previous_date_features)
    # Predict lag value for the current previous_date
        previous_date_lag = ridge_regression_predict(coefficients, np.c_[np.ones(previous_date_features_scaled.shape[0]), previous_date_features_scaled])[0] 
    # Update previous_date to the next day
        previous_date += pd.DateOffset(days=1)

        # Update previous_date_features with the predicted lag value and date features
        previous_date_features = pd.DataFrame({
            'day': [int(previous_date.day)],
            'month': [int(previous_date.month)],
            'year': [int(previous_date.year)],
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
            'Average_lag1': [previous_date_lag],
        })

        previous_date_features = previous_date_features.values.reshape(1, -1)

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
            'Average_lag1' : [previous_date_lag],
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


def show_result(request, commodity, predicted_price):
    context = {
        'commodity': commodity,
        'predicted_price': predicted_price,
    }
    return render(request, 'result.html', context)

def analysis_view(request):
    return render(request, 'analysis.html')

def analysis_fig(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_commodity = data.get('commodity')
        if not selected_commodity:
            return JsonResponse({'error': 'No commodity selected'})

        df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\lagadded_out.csv')
        df1 = df[df['Commodity'] == selected_commodity]
        if df1.empty:
            return JsonResponse({'error': f'No data found for {selected_commodity}'})

        df1['Date'] = pd.to_datetime(df1['Date'])

        df1 = df1.sort_values('Date')
        df1 = df1[(df1['Date'] < '2020-01-01') | (df1['Date'] >= '2021-01-01')]
        latest_months = 6

        latest_date = df1['Date'].max()
        earliest_date = latest_date - pd.DateOffset(months=latest_months)

        train_data = df1[df1['Date'] < earliest_date]
        test_data = df1[df1['Date'] >= earliest_date]

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

        # Train the Ridge Regression model
        coefficients = ridge_regression_fit(X_train, y_train, alpha=1.0)

        # Make predictions
        y_pred = ridge_regression_predict(coefficients, X_test)
        predicted_prices = list(np.exp(y_pred))

        # Filter actual prices for the last 6 months
        actual_prices_data = df1[df1['Date'] >= earliest_date]
        actual_prices = list(np.exp(actual_prices_data['Average']))

        season_columns = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']
        festival_columns = [col for col in df1.columns if 'Festival_' in col and '_near' not in col]

        # Get lag columns related to Seasons and Festivals
        lag_season_columns = ['Fall_near','Spring_near','Summer_near','Winter_near']
        lag_festival_columns = ['Dashain_near', 'Tihar_near', 'Holi_near', 'Maha Shivaratri_near', 'Buddha Jayanti_near', 'Ghode Jatra_near', 'Teej_near', 'Indra Jatra_near', 'Lhosar_near', 'Janai Purnima_near', 'Gai Jatra_near', 'Maghe Sankranti_near', 'Shree Panchami_near']

        actual_vs_predicted_buffer = io.BytesIO()
        seasons_buffer = io.BytesIO()
        festivals_buffer = io.BytesIO()

        plt.figure(figsize=(24, 6))
        plt.subplot(1, 3, 1)
        plt.plot(actual_prices_data['Date'], actual_prices, label='Actual Prices', marker='o')
        plt.plot(test_data['Date'], predicted_prices, label='Predicted Prices', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Actual vs Predicted Prices - {selected_commodity}')
        plt.legend()
        plt.savefig(actual_vs_predicted_buffer, format='png')
        plt.close()

        # Plot Price changes with Seasons and their lag features
        plt.figure(figsize=(22, 8))
        plt.subplot(1, 3, 2)
        for season_col, lag_season_col in zip(season_columns, lag_season_columns):
            avg_prices = df1[df1['Date'].dt.year == 2023][f'{season_col}'] + df1[df1['Date'].dt.year == 2023][lag_season_col]
            plt.plot(season_col.split('_')[-1], np.mean(avg_prices), alpha=0.7)

        plt.xlabel('Season')
        plt.ylabel('Average Transformed Price')
        plt.title(f'Average Price changes with Seasons - {selected_commodity}')
        plt.savefig(seasons_buffer, format='png')
        plt.close()

        plt.figure(figsize=(22, 8))
        plt.subplot(1, 3, 3)
        for festival_col, lag_festival_col in zip(festival_columns, lag_festival_columns):
            avg_prices = df1[df1['Date'].dt.year == 2023][festival_col] + df1[df1['Date'].dt.year == 2023][lag_festival_col]
            plt.plot(festival_col.split('_')[-1], np.mean(avg_prices))

        plt.xlabel('Festival')
        plt.ylabel('Average Transformed Price')
        plt.title(f'Average Price changes with Festivals - {selected_commodity}')
        plt.savefig(festivals_buffer, format='png')
        plt.close()
        # Save the plots as separate image files
        # actual_vs_predicted_path = os.path.join(settings.MEDIA_ROOT, 'actual_vs_predicted_plot.png')
        # seasons_path = os.path.join(settings.MEDIA_ROOT, 'seasons_plot.png')
        # festivals_path = os.path.join(settings.MEDIA_ROOT, 'festivals_plot.png')

        # plt.savefig(actual_vs_predicted_path, format='png')
        # plt.savefig(seasons_path, format='png')
        # plt.savefig(festivals_path, format='png')

        # plt.close()
        actual_vs_predicted_base64 = base64.b64encode(actual_vs_predicted_buffer.getvalue()).decode('utf-8')
        seasons_base64 = base64.b64encode(seasons_buffer.getvalue()).decode('utf-8')
        festivals_base64 = base64.b64encode(festivals_buffer.getvalue()).decode('utf-8')
        context = {
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices,
            'commodity' : selected_commodity,
            'actual_vs_predicted_base64': actual_vs_predicted_base64,
            'seasons_base64': seasons_base64,
            'festivals_base64': festivals_base64,

        }

        return JsonResponse(context)
    else:
        return JsonResponse({'error': 'Invalid request method'})
