
        selected_date = datetime.strptime(selected_date, '%Y-%m-%d')
        last_date = df['Date'].max()
        days_difference = (selected_date - last_date).days
        predictions_df = pd.DataFrame(columns=['Commodity', 'Date', 'PredictedPrice'])

        for commodity in df['Commodity'].unique():
            # Similar logic as before for each commodity
            last_commodity_date = df[df['Commodity'] == commodity]['Date'].max()
            commodity_data = df[df['Commodity'] == commodity].tail(1).drop(['Average', 'Commodity', 'Date'], axis=1)
    
            # Predicting for a future date based on the user input (calculated days difference)
            future_date = last_commodity_date + timedelta(days=days_difference)
            predicted_log_price = lmodel.predict(commodity_data)[0]
            predicted_price = np.exp(predicted_log_price)
    
            # Append the prediction to the DataFrame
            predictions_df = pd.concat([predictions_df, pd.DataFrame({'Commodity': [commodity],
                                                                      'Date': [future_date],
                                                                      'PredictedPrice': [predicted_price]})])

            return JsonResponse({'predicted_prices': predictions_df.to_dict('records')})
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

        df = pd.read_csv(r'C:\Users\nirvi\OneDrive\Desktop\Programs\Mlcurrent\train1.csv')
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