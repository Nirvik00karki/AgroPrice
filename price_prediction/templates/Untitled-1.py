
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