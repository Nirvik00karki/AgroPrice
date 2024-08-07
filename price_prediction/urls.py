from django.urls import path,include
from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.conf.urls.static import static
from . import views
# from .views import HistoricalPriceViewSet


router = DefaultRouter()
# router.register(r'PredictionModel', HistoricalPriceViewSet, basename='predictionmodel')

urlpatterns = [
    path('home/', views.home, name='home'),
    path('', views.index, name='index'),
    path('api/', include(router.urls)),
    # path('api/serve_overall_table/', views.serve_overall_table, name='serve_overall_table'),
    path('prices/', views.render_price_search, name='render_price_search'),
    path('prices/data/', views.get_price_data, name='get_price_data'),
    path('account/', views.account, name='account'),
    path('login/', views.user_login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('check_email/', views.check_email, name='check_email'),
    path('submit_review/', views.submit_review, name='submit_review'),
    path('home/commodity_detail/', views.commodity_detail, name='commodity_detail'),
    path('api/predict/', views.predict_average_price, name='predict_average_price'),
    path('get_commodities/', views.get_commodities, name='get_commodities'),
    path('get_commodity_data/<path:commodity_names>/', views.get_commodity_data, name='get_commodity_data'),
    path('search_commodity/', views.search_commodity, name='search_commodity'),
    path('result/<str:commodity>/<path:predicted_price>/', views.show_result, name='show_result'),
    path('potato/', views.potato_detail, name='potato'),
    path('api/analyze/', views.analysis_fig, name='analysis_fig'),
    path('analysis_view/', views.analysis_view, name='analysis_view'),
    path('analysis_view2/', views.analysis_view2, name='analysis_view2'),
    path('generate_chart/', views.generate_chart, name='generate_chart'),
    # path('analysis/<str:selected_commodity>/', views.render_analysis, name='render_analysis'),
  ]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)