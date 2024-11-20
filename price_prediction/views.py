from django.shortcuts import render,redirect, HttpResponse
from django.contrib.auth.decorators import login_required
# from rest_framework import viewsets
# from .models import HistoricalPrice
# from .serializers import HistoricalPriceSerializer
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.pagination import PageNumberPagination
# from sklearn.linear_model import Ridge
# from statsmodels.tsa.arima.model import ARIMA
from .models import CustomUser
# from sklearn.ensemble import GradientBoostingRegressor
from django.http import JsonResponse
import pandas as pd
import numpy as np
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from .models import Review 
from sklearn.model_selection import train_test_split

@login_required
def home(request):
    return render(request, 'home.html', {'user': request.user})

def index(request):
    return render(request, 'index.html')

def potato_detail(request):
    return render(request, 'potato.html')

def commodity_detail(request):
    return render(request, 'commodity_detail.html')

def render_price_search(request):
    return render(request, 'pricesearch.html')

# View function to return the CSV data
def get_price_data(request):
    df = pd.read_csv(r'datasets\Pricestoday.csv')
    data = df.to_dict(orient='records')
    return JsonResponse(data, safe=False)
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

def check_email(request):
    if request.method == 'POST':
        email = request.POST.get('email', None)
        if email:
            exists = CustomUser.objects.filter(email=email).exists()
            return JsonResponse({'exists': exists})
    return JsonResponse({'error': 'Invalid request'})
    
def get_commodities(request):
    df = pd.read_csv(r'datasets\first_merged_1.csv')
    commodities = df['Commodity'].unique().tolist()
    return JsonResponse(commodities, safe=False)

def get_commodity_data(request, commodity_names):
    commodity_names = json.loads(commodity_names)
    df = pd.read_csv(r'datasets\first_merged_1.csv')

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
    df = pd.read_csv(r'datasets\first_merged_1.csv')
    search_results = df['Commodity'][df['Commodity'].str.contains(term, case=False)].unique().tolist()
    return JsonResponse(search_results, safe=False)


def ridge_regression_fit(X, y, alpha=7.0, fit_intercept=True):
    if fit_intercept:
        X = np.c_[np.ones(X.shape[0]), X] 

    n_features = X.shape[1]
    identity_matrix = np.identity(n_features)
    coefficients = np.linalg.inv(X.T @ X + alpha * identity_matrix) @ X.T @ y
    print(coefficients)
    return coefficients

def ridge_regression_predict(coefficients, X, fit_intercept=True):
    if fit_intercept:
        X = np.c_[np.ones(X.shape[0]), X] 

    predictions = X @ coefficients
    print(predictions)
    return predictions

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

        df = pd.read_csv(r'datasets\train1.csv')
        # df.drop(columns=['Average'], inplace=True)
        df = df[df['Commodity'] == selected_commodity]
        if df.empty:
            return JsonResponse({'error': f'No data found for {selected_commodity}'})
     
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # df1 = df1.sort_values('Date')
        # df1 = df1[(df1['Date'] < '2020-01-01') | (df1['Date'] >= '2021-01-01')]
        # latest_months = 6 
        # latest_date = df1['Date'].max()
        # earliest_date = latest_date - pd.DateOffset(months=latest_months)
        # train_data = df1[df1['Date'] < earliest_date]
        # test_data = df1[df1['Date'] >= earliest_date]
        x = df.drop(['Spline_Average', 'Commodity', 'Date'], axis=1)
        y = df['Spline_Average']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        X_train = np.c_[np.ones(x_train_scaled.shape[0]), x_train_scaled]
        X_test = np.c_[np.ones(x_test_scaled.shape[0]), x_test_scaled]
        print("Shapes before training - X_train:", X_train.shape, "X_test:", X_test.shape)

        coefficients = ridge_regression_fit(X_train, y_train, alpha=7.0)
        print("Shapes before prediction - X_test:", X_test.shape)
        y_pred = ridge_regression_predict(coefficients, X_test)
        # gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        try:
            selected_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
        except ValueError:
            return JsonResponse({'error': 'Invalid date format'})
        # known_lag_date = df1['Date'].iloc[-1]
        last_date = df['Date'].max()
        selected_commodity_lag = df['Spline_Average'].iloc[-1]
        known_lag_value = selected_commodity_lag
        last_ma = df['MA'].iloc[-1]
        last_ema = df['EMA'].iloc[-1]
        previous_date = last_date + pd.DateOffset(days=1)
        # week_sin = np.sin(2 * np.pi * previous_date.isocalendar().week / 52)
        # week_cos = np.cos(2 * np.pi * previous_date.isocalendar().week / 52)
        # day_sin = np.sin(2 * np.pi * previous_date.day / 365)
        # day_cos = np.cos(2 * np.pi * previous_date.day / 365)
        previous_date_features = pd.DataFrame({
            # 'day': [int(previous_date.day)],
            'month': [int(previous_date.month)],
            'year': [int(previous_date.year)],
            'Season_Fall': [0], 
            'Season_Spring': [1],
            'Season_Summer': [0],
            'Season_Winter': [0],
            'Apple_Jholey': [1],
            'Banana': [1],
            'Carrot_Local': [1],
            'Cucumber_Local': [1],
            'Garlic_Dry_Nepali': [1],
            'Lettuce': [1],
            'Onion_Dry_Indian': [1],
            'Potato_White': [1],
            'Tomato_Big_Nepali': [1],
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
            'Festival_nan': [1],
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
            'Spring_near': [1],
            'Summer_near': [0],
            'Winter_near': [0],
            'Spline_Average_Lag1': [known_lag_value],
            'MA': [last_ma],
            'EMA': [last_ema],
            'day': [int(previous_date.day)],
            'Week': [int(previous_date.isocalendar().week)],
            # 'week': [previous_date.isocalendar().week],
            # 'Week_sin':[week_sin],
            # 'Week_cos':[week_cos],
        })

        previous_date_features = previous_date_features.values.reshape(1, -1)
        while previous_date.date() < selected_date:
            # if not df1[df1['Date'] == previous_date.date()].empty:
            previous_date_features_scaled = scaler.transform(previous_date_features)
            previous_date_lag = ridge_regression_predict(coefficients, np.c_[np.ones(previous_date_features_scaled.shape[0]), previous_date_features_scaled])[0] 
            # previous_date_lag = gb.predict(previous_date_features_scaled)[0]
            ma_window = 7
            ma_value = df[df['Commodity'] == selected_commodity]['Spline_Average'].rolling(window=ma_window, min_periods=1).mean().iloc[-1]

            ema_span = 15 
            ema_value = df[df['Commodity'] == selected_commodity]['Spline_Average'].ewm(span=ema_span, adjust=False).mean().iloc[-1]

            previous_date += pd.DateOffset(days=1)
            print("Lag Prediction:", previous_date_lag)
            previous_date_features = pd.DataFrame({
                # 'day': [int(previous_date.day)],
                'month': [int(previous_date.month)],
                'year': [int(previous_date.year)],
                'Season_Fall': [0], 
                'Season_Spring': [1],
                'Season_Summer': [0],
                'Season_Winter': [0],
                'Apple_Jholey': [1],
                'Banana': [1],
                'Carrot_Local': [1],
                'Cucumber_Local': [1],
                'Garlic_Dry_Nepali': [1],
                'Lettuce': [1],
                'Onion_Dry_Indian': [1],
                'Potato_White': [1],
                'Tomato_Big_Nepali': [1],
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
                'Festival_nan': [1],
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
                'Spring_near': [1],
                'Summer_near': [0],
                'Winter_near': [0],
                'Spline_Average_Lag1': [previous_date_lag],
                'MA': [ma_value],
                'EMA': [ema_value],
                'day': [int(previous_date.day)],
                'Week': [int(previous_date.isocalendar().week)],
            })

            previous_date_features = previous_date_features.values.reshape(1, -1)

        selected_date_features = pd.DataFrame({
            # 'day': [int(selected_date.day)],
            'month': [int(selected_date.month)],
            'year': [int(selected_date.year)],
            'Season_Fall': [0], 
            'Season_Spring': [1],
            'Season_Summer': [0],
            'Season_Winter': [0],
            'Apple_Jholey': [1],
            'Banana': [1],
            'Carrot_Local': [1],
            'Cucumber_Local': [1],
            'Garlic_Dry_Nepali': [1],
            'Lettuce': [1],
            'Onion_Dry_Indian': [1],
            'Potato_White': [1],
            'Tomato_Big_Nepali': [1],
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
            'Festival_nan': [1],
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
            'Spring_near': [1],
            'Summer_near': [0],
            'Winter_near': [0],
            'Spline_Average_Lag1': [previous_date_lag],
            'MA': [ma_value],
            'EMA': [ema_value],
            'day': [int(selected_date.day)],
            'Week': [int(selected_date.isocalendar().week)],
        })

        selected_date_features = selected_date_features.values.reshape(1, -1)
        print("Selected Date Features:", selected_date_features)
        selected_date_features_scaled = scaler.transform(selected_date_features)
        selected_date_features_scaled_with_bias = np.c_[np.ones(selected_date_features_scaled.shape[0]), selected_date_features_scaled]
        # selected_date_features_with_bias = np.c_[np.ones(selected_date_features.shape[0]), selected_date_features]
        # print("Selected Date Features with Bias:", selected_date_features_scaled_with_bias)
        prediction = ridge_regression_predict(coefficients, selected_date_features_scaled_with_bias)
        # prediction = gb.predict(selected_date_features_scaled)
        print("Raw Prediction:", prediction)
        predicted_average_price = prediction[0]
        print("Predicted Price:", predicted_average_price)
        context = {
            'predicted_price': predicted_average_price,
            # 'predicted_prices': json.dumps(list(np.exp(y_pred))),
            #'actual_prices': json.dumps(actual_prices),
            'mse': mse,
            'r2': r2,
            'date': selected_date,
            'commodity': selected_commodity,
        }
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

def analysis_view2(request):
    return render(request, 'analysis2.html')

def analysis_fig(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_commodity = data.get('commodity')
        if not selected_commodity:
            return JsonResponse({'error': 'No commodity selected'})
        context = {

            'commodity' : selected_commodity,

        }

        return JsonResponse(context)
    else:
        return JsonResponse({'error': 'Invalid request method'})

def generate_chart(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_commodity = data.get('commodity')
        if not selected_commodity:
            return JsonResponse({'error': 'No commodity selected'})

        df = pd.read_csv(r'datasets\forchart2.csv')
        df1 = df[df['Commodity'] == selected_commodity]
        if df1.empty:
            return JsonResponse({'error': f'No data found for {selected_commodity}'})
        df1['Date'] = pd.to_datetime(df1['Date'])
        df1['Average'] = np.exp(df1['Average'])
        df_2023_2024 = df1[(df1['Date'].dt.year == 2023) | (df1['Date'].dt.year == 2024)]
        seasons = df_2023_2024['Season'].tolist()
        averages = df_2023_2024['Average'].tolist()
        dates = df_2023_2024['Date'].tolist()
        festivals = df_2023_2024['Festival'].tolist()
        years = df1['year'].tolist()
        yearAverages = df1['Average'].tolist()
        
        context = {'seasons': seasons, 'averages': averages, 'dates': dates, 'years': years, 'yearAverages': yearAverages, 'festivals': festivals }
        return JsonResponse(context)
    else:
        return JsonResponse({'error': 'Invalid request method'})
 
def submit_review(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        review = Review.objects.create(name=name, email=email, message=message)
        review.save()
        review_saved = True
    return render(request, 'index.html', {'review_saved': review_saved})