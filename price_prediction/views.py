from django.shortcuts import render,redirect,get_object_or_404
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
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import CustomUser
import joblib 
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from .models import Potato_white
from sklearn.metrics import mean_squared_error, r2_score

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

def serve_overall_table(request):
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


def train_model(request):
    commodity_data = Potato_white.objects.all().values('Commodity', 'Date', 'Average', 'day', 'month', 'year', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Apple_Jholey', 'Banana', 'Carrot_Local', 'Cucumber_Local', 'Garlic_Dry_Nepali', 'Lettuce', 'Onion_Dry_Indian', 'Potato_White', 'Tomato_Big_Nepali', 'Festival_Buddha_Jayanti', 'Festival_Dashain', 'Festival_Gai_Jatra', 'Festival_Ghode_Jatra', 'Festival_Holi', 'Festival_Indra_Jatra', 'Festival_Janai_Purnima', 'Festival_Lhosar', 'Festival_Maghe_Sankranti', 'Festival_Maha_Shivaratri', 'Festival_Shree_Panchami', 'Festival_Teej', 'Festival_Tihar', 'Festival_nan', 'Dashain_near', 'Tihar_near', 'Holi_near', 'Maha_Shivaratri_near', 'Buddha_Jayanti_near', 'Ghode_Jatra_near', 'Teej_near', 'Indra_Jatra_near', 'Lhosar_near', 'Janai_Purnima_near', 'Gai_Jatra_near', 'Maghe_Sankranti_near', 'Shree_Panchami_near', 'Fall_near', 'Spring_near', 'Summer_near', 'Winter_near')

    df = pd.DataFrame(commodity_data)

    X = df.drop(['Average', 'Commodity', 'Date'], axis=1)
    y = df['Average']
    lmodel = LinearRegression()
    lmodel.fit(X, y)

    y_pred = lmodel.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print(f'R-squared: {r2}')
    print(f'Mean Squared Error: {mse}')

    df['Date'] = pd.to_datetime(df['Date'])

    predictions_df = pd.DataFrame(columns=['Commodity', 'Date', 'PredictedPrice'])

    for commodity in df['Commodity'].unique():
        last_date = df[df['Commodity'] == commodity]['Date'].max()
        commodity_data = df[df['Commodity'] == commodity].tail(1).drop(['Average', 'Commodity', 'Date'], axis=1)
        predicted_log_price = lmodel.predict(commodity_data)[0]
        predicted_price = np.exp(predicted_log_price)
        predictions_df = pd.concat([predictions_df, pd.DataFrame({'Commodity': [commodity],
                                                              'Date': [last_date + timedelta(days=7)],
                                                              'PredictedPrice': [predicted_price]})])
 
    return JsonResponse({
            'message': 'Model training completed',
            'R2 Score': r2,
            'MSE': mse,
            'PredictedPriceNextWeek': predicted_price
        })