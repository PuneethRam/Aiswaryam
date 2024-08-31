from django.shortcuts import render, redirect
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout
import os
import matplotlib.pyplot as plt
import io
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.template.loader import render_to_string
import pandas as pd
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
from .models import AnalysisResult
from .models import TelegramAnalysis
from ultralytics import YOLO
from IPython import display
display.clear_output()
import ultralytics
import folium
from folium.plugins import MarkerCluster
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from django.shortcuts import redirect
from django.shortcuts import redirect, render
import datetime


def index(request):
  context = {
    'segment': 'index'
  }
  return render(request, "pages/index.html", context)

def tables(request):
  context = {
    'segment': 'tables'
  }
  return render(request, "pages/tables.html", context)

# Components
@login_required(login_url='/accounts/login/')
def bc_button(request):
  context = {
    'parent': 'basic_components',
    'segment': 'button'
  }
  return render(request, "pages/components/bc_button.html", context)

@login_required(login_url='/accounts/login/')
def bc_badges(request):
  context = {
    'parent': 'basic_components',
    'segment': 'badges'
  }
  return render(request, "pages/components/bc_badges.html", context)

@login_required(login_url='/accounts/login/')
def bc_breadcrumb_pagination(request):
  context = {
    'parent': 'basic_components',
    'segment': 'breadcrumbs_&_pagination'
  }
  return render(request, "pages/components/bc_breadcrumb-pagination.html", context)

@login_required(login_url='/accounts/login/')
def bc_collapse(request):
  context = {
    'parent': 'basic_components',
    'segment': 'collapse'
  }
  return render(request, "pages/components/bc_collapse.html", context)

@login_required(login_url='/accounts/login/')
def bc_tabs(request):
  context = {
    'parent': 'basic_components',
    'segment': 'navs_&_tabs'
  }
  return render(request, "pages/components/bc_tabs.html", context)

@login_required(login_url='/accounts/login/')
def bc_typography(request):
  context = {
    'parent': 'basic_components',
    'segment': 'typography'
  }
  return render(request, "pages/components/bc_typography.html", context)

@login_required(login_url='/accounts/login/')
def icon_feather(request):
  context = {
    'parent': 'basic_components',
    'segment': 'feather_icon'
  }
  return render(request, "pages/components/icon-feather.html", context)


# Forms and Tables
@login_required(login_url='/accounts/login/')
def form_elements(request):
  context = {
    'parent': 'form_components',
    'segment': 'form_elements'
  }
  return render(request, 'pages/form_elements.html', context)

@login_required(login_url='/accounts/login/')
def basic_tables(request):
  context = {
    'parent': 'tables',
    'segment': 'basic_tables'
  }
  return render(request, 'pages/tbl_bootstrap.html', context)

# Chart and Maps
@login_required(login_url='/accounts/login/')
def morris_chart(request):
  context = {
    'parent': 'chart',
    'segment': 'morris_chart'
  }
  return render(request, 'pages/chart-morris.html', context)

@login_required(login_url='/accounts/login/')
def google_maps(request):
  context = {
    'parent': 'maps',
    'segment': 'google_maps'
  }
  return render(request, 'pages/map-google.html', context)

# Authentication
class UserRegistrationView(CreateView):
  template_name = 'accounts/auth-signup.html'
  form_class = RegistrationForm
  success_url = '/accounts/login/'

class UserLoginView(LoginView):
  template_name = 'accounts/auth-signin.html'
  form_class = LoginForm

class UserPasswordResetView(PasswordResetView):
  template_name = 'accounts/auth-reset-password.html'
  form_class = UserPasswordResetForm

class UserPasswrodResetConfirmView(PasswordResetConfirmView):
  template_name = 'accounts/auth-password-reset-confirm.html'
  form_class = UserSetPasswordForm

class UserPasswordChangeView(PasswordChangeView):
  template_name = 'accounts/auth-change-password.html'
  form_class = UserPasswordChangeForm

def logout_view(request):
  logout(request)
  return redirect('/accounts/login/')

@login_required(login_url='/accounts/login/')
def profile(request):
  context = {
    'segment': 'profile',
  }
  return render(request, 'pages/profile.html', context)

@login_required(login_url='/accounts/login/')
def sample_page(request):
  context = {
    'segment': 'sample_page',
  }
  return render(request, 'pages/sample-page.html', context)



@login_required(login_url='/accounts/login/')
def pedestrian(request):
  context = {
    'segment': 'sample_page',
  }
  return render(request, 'pages/pedestrian.html', context)



@login_required(login_url='/accounts/login/')
def overall(request):
    map_html_path = '' 

    df=pd.read_csv(r'home\datasets\FINAL_KARNATAKA_DATA.csv')

    if request.method == 'POST':
        district = request.POST.get('district')
        districts_to_keep = [district]
        df = df[df['DISTRICTNAME'].isin(districts_to_keep)]
        center_point = [12.9716, 77.5946]  # Example center point for Mysuru City
        m = folium.Map(location=center_point, zoom_start=7)
        marker_cluster = MarkerCluster().add_to(m)
        for index, row in df.iterrows():
            # Create a popup message for the marker
            popup_text = f"Accident Spot: {row['Accident_Spot']}\nAccident SubLocation: {row['Accident_SubLocation']}\nSeverity: {row['Severity']}"

            # Create a marker and add it to the MarkerCluster layer
            folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text).add_to(marker_cluster)
        # Save the map as an HTML file
        map_html_path = r'static\assets\maps\maps.html'
        m.save(map_html_path)


    

    accident_analysis = []
    for year in range(2016, 2024):
        fatal_count = df[(df['Year'].astype(int) == year) & (df['Severity'].astype(str) == 'Fatal')].shape[0]
        non_fatal_count = df[(df['Year'].astype(int) == year) & (df['Severity'].astype(str) != 'Fatal')].shape[0]
        accident_analysis.append({'year': year, 'fatal': fatal_count, 'non_fatal': non_fatal_count})
    main_cause_counts = df['Main_Cause'].value_counts().reset_index().rename(columns={'index': 'Main_Cause', 'Main_Cause': 'count'})


    year_roadcondition_counts = df.groupby(['Year', 'Road_Condition']).size().reset_index(name='count')

    data_by_year = {}
    for _, row in year_roadcondition_counts.iterrows():
        year = row['Year']
        roadcondition = row['Road_Condition']  # Corrected column name
        count = row['count']
        if year not in data_by_year:
            data_by_year[year] = {}
        data_by_year[year][roadcondition] = count
    years = sorted(data_by_year.keys())
    roadconditions = sorted(list(set(rc for rc_counts in data_by_year.values() for rc in rc_counts.keys())))
    series = [{ 'name': rc, 'data': [data_by_year.get(year, {}).get(rc, 0) for year in years] } for rc in roadconditions]


#---------------------------------------------------------------------------------------------------------------------------
    #weather severity
    # Group by 'Weather' and 'Severity' and count occurrences
    weather_severity_counts = df.groupby(['Weather', 'Severity']).size().reset_index(name='count')

    # Prepare data for Highcharts
    weather_categories = sorted(df['Weather'].unique())
    fatal_data = []
    non_fatal_data = []

    for weather in weather_categories:
        fatal_count = weather_severity_counts[(weather_severity_counts['Weather'] == weather) & (weather_severity_counts['Severity'] == 'Fatal')]['count'].sum()
        non_fatal_count = weather_severity_counts[(weather_severity_counts['Weather'] == weather) & (weather_severity_counts['Severity'] != 'Fatal')]['count'].sum()
        fatal_data.append(fatal_count)
        non_fatal_data.append(non_fatal_count)

    series_data = [
        {
            'name': 'Fatal',
            'data': fatal_data
        },
        {
            'name': 'Non Fatal',
            'data': non_fatal_data
        }
    ]




#--------------------------------------------------------------------------------------------------------------------
#accident spot
    accident_collision_counts = df.groupby(['Accident_Spot', 'Severity']).size().reset_index(name='count')
    
    # Prepare data for Highcharts
    accident_spots = sorted(df['Accident_Spot'].unique())
    collision_types = sorted(df['Severity'].unique())
    series_data2 = []

    for collision_type in collision_types:
        data = []
        for accident_spot in accident_spots:
            count = accident_collision_counts[(accident_collision_counts['Accident_Spot'] == accident_spot) & (accident_collision_counts['Severity'] == collision_type)]['count'].sum()
            data.append(count)
        series_data2.append({
            'name': collision_type,
            'data': data
        })    




     
#---------------------------------------------------------------------------------------------------------------------
    context = {
       'accident_analysis': accident_analysis,
       'main_cause_counts': main_cause_counts.to_dict('records'),
       'years': years,
       'series': series,
       'weather_categories': weather_categories,
       'series_data': series_data,
       'map_html_path': map_html_path,
       'accident_spots': accident_spots,
       'collision_types': collision_types,
       'series_data2': series_data2,
    }

    return render(request, 'pages/overall.html', context)




CLIENT_SECRETS_FILE = 'home\client_secret_1068510412293-2n0v2a8vm2lbajlronahqn3i4r294869.apps.googleusercontent.com.json'
SCOPES = ['https://www.googleapis.com/auth/bigquery.readonly']

@login_required(login_url='/accounts/login/')
def timeanalysis(request): 
    map_html_path = ''
    if request.method == 'POST':
        day = request.POST.get('day')
        start_date = pd.to_datetime(request.POST.get('startdate'))
        end_date= pd.to_datetime(request.POST.get('enddate'))
        starttime = pd.to_datetime(request.POST.get('starttime')).time()
        endtime = pd.to_datetime(request.POST.get('endtime')).time()
        print(start_date,end_date,day,starttime,endtime)
        df=pd.read_csv(r'home\datasets\pollution_air.csv')
        df['local_time'] = pd.to_datetime(df['local_time'])
        df_date = df[(df['local_time'] >= start_date) & (df['local_time'] <= end_date)]
        print(len(df_date))
        day = [day]
        df_day = df_date[df_date['day'].isin(day)]
        
        
        df_time = df_day[df_day['local_time'].dt.time.between(starttime, endtime)]
        
        train_data = df_time.drop(columns=['local_time','thing_id'	, 'latitude', 'longitude'])

        # Identify categorical and numerical columns
        numerical_cols =  ['temperature','humidity','pm2_5','pm10' ,'co2',	'no2','so2','tvoc','o3','co','no','nh3']

        numerical_transformer = StandardScaler()

        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
            ]
        )

        # Define the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', IsolationForest(contamination=0.1))
        ])

        pipeline.fit(train_data)
        anomalies = pipeline.predict(train_data)

        df_time['anomaly'] = anomalies
        df_time['anomaly'] = df_time['anomaly'].apply(lambda x: 1 if x == -1 else 0) 

        center_point = [13.0524, 80.1612]  
        location_counts = df_time[df_time['anomaly'] == 1].groupby(['latitude', 'longitude']).size().reset_index(name='count')

        # Create a folium map
        m = folium.Map(location=center_point, zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)

        # Add circle markers to the map
        for index, row in location_counts.iterrows():
            lat, lon, count = row['latitude'], row['longitude'], row['count']
            popup_text = f'Occurrences: {count}'
            folium.CircleMarker(
                location=[lat, lon],
                popup=popup_text,  # Assigning the popup content to the marker
                radius= 5 + count//2,  # Adjust the multiplier as needed
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)

        # Save the map to an HTML file
        map_html_path = r'static\assets\maps\airpollution.html'
        m.save(map_html_path)


    return render(request, 'pages/timeanalysis.html', {'map_html_path': map_html_path})



@api_view(['GET'])
def telegram(request):
    lat = float(request.GET.get('lat'))
    lon = float(request.GET.get('lon'))
    df=pd.read_csv(r'home\datasets\updated_dataset.csv')
    RegionData = df[(df['Region_Latitude'].astype(float) == lat) & (df['Region_Longitude'].astype(float) == lon)]
    
    

    accident_analysis = []
    for year in range(2016, 2024):
        fatal_count = RegionData[(RegionData['Year'].astype(int) == year) & (RegionData['Severity'].astype(str) == 'Fatal')].shape[0]
        non_fatal_count = RegionData[(RegionData['Year'].astype(int) == year) & (RegionData['Severity'].astype(str) != 'Fatal')].shape[0]
        accident_analysis.append({'year': year, 'fatal': fatal_count, 'non_fatal': non_fatal_count})
    main_cause_counts = RegionData['Main_Cause'].value_counts().reset_index().rename(columns={'index': 'Main_Cause', 'Main_Cause': 'count'})
    junction_control_counts = RegionData['Junction_Control'].value_counts().reset_index().rename(columns={'index': 'Junction_Control', 'Junction_Control': 'count'})
    print(main_cause_counts)
#----------------------------------------------------------------------------------------------------
    #Road condition

    year_roadcondition_counts = RegionData.groupby(['Year', 'Road_Condition']).size().reset_index(name='count')

    data_by_year = {}
    for _, row in year_roadcondition_counts.iterrows():
        year = row['Year']
        roadcondition = row['Road_Condition']  # Corrected column name
        count = row['count']
        if year not in data_by_year:
            data_by_year[year] = {}
        data_by_year[year][roadcondition] = count
    years = sorted(data_by_year.keys())
    roadconditions = sorted(list(set(rc for rc_counts in data_by_year.values() for rc in rc_counts.keys())))
    series = [{ 'name': rc, 'data': [data_by_year.get(year, {}).get(rc, 0) for year in years] } for rc in roadconditions]


#---------------------------------------------------------------------------------------------------------------------------
    #weather severity
    # Group by 'Weather' and 'Severity' and count occurrences
    weather_severity_counts = RegionData.groupby(['Weather', 'Severity']).size().reset_index(name='count')

    # Prepare data for Highcharts
    weather_categories = sorted(RegionData['Weather'].unique())
    fatal_data = []
    non_fatal_data = []

    for weather in weather_categories:
        fatal_count = weather_severity_counts[(weather_severity_counts['Weather'] == weather) & (weather_severity_counts['Severity'] == 'Fatal')]['count'].sum()
        non_fatal_count = weather_severity_counts[(weather_severity_counts['Weather'] == weather) & (weather_severity_counts['Severity'] != 'Fatal')]['count'].sum()
        fatal_data.append(fatal_count)
        non_fatal_data.append(non_fatal_count)

    series_data = [
        {
            'name': 'Fatal',
            'data': fatal_data
        },
        {
            'name': 'Non Fatal',
            'data': non_fatal_data
        }
    ]


#--------------------------------------------------------------------------------------------------------------------
#accident spot
    accident_collision_counts = RegionData.groupby(['Accident_Spot', 'Collision_Type']).size().reset_index(name='count')
    
    # Prepare data for Highcharts
    accident_spots = sorted(RegionData['Accident_Spot'].unique())
    collision_types = sorted(RegionData['Collision_Type'].unique())
    series_data2 = []

    for collision_type in collision_types:
        data = []
        for accident_spot in accident_spots:
            count = accident_collision_counts[(accident_collision_counts['Accident_Spot'] == accident_spot) & (accident_collision_counts['Collision_Type'] == collision_type)]['count'].sum()
            data.append(count)
        series_data2.append({
            'name': collision_type,
            'data': data
        })   
    

#---------------------------------------------------------------------------------------------------------------------
    context = {
       'accident_analysis': accident_analysis,
       'main_cause_counts': main_cause_counts.to_dict('records'),
       'junction_control_counts':junction_control_counts.to_dict('records'),
       'years': years,
       'series': series,
       'weather_categories': weather_categories,
       'series_data': series_data,
       'accident_spots': accident_spots,
       'collision_types': collision_types,
       'series_data2': series_data2,
    
    }

    return Response(context)
    
    

@api_view(['GET'])
def insights(request):
    lat = float(request.GET.get('lat'))
    lon = float(request.GET.get('lon'))
    df=pd.read_csv(r'home\datasets\updated_dataset.csv')
    RegionData = df[(df['Region_Latitude'].astype(float) == lat) & (df['Region_Longitude'].astype(float) == lon)]  
    columns_to_keep = ['Main_Cause', 'Severity', 'Collision_Type','Junction_Control','Surface_Condition','Weather']
    new_df = RegionData[columns_to_keep]

    # Convert the dataframe to a string format
    df_string = new_df.to_string(index=False)

    import google.generativeai as genai
    genai.configure(api_key="AIzaSyD01Eqx2S8qwI0NrsFztOkHiDqpBMCUGvA")   
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"I am analyzing the accidents that happened over a particular region. I have attached the dataframe below which contains some parameters regarding the accidents that occurred in that area. Please analyze it and provide an analysis including insights, major contributing factors, and any other relevant information you can infer. Keep the response short. Dataframe:\n{df_string}")
    insights=response.text


#---------------------------------------------------------------------------------------------------------------------
    context = {
       'insights' : insights,
    }

    return Response(context)   

@api_view(['POST'])
def blackspots(request):
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import IsolationForest
        map_html_path = '' 
        
        data = request.data
        print(data)
        # Extract parameters
        district = data.get('district')
        starttime = pd.to_datetime(data.get('starttime'), errors='coerce').time() if data.get('starttime') else datetime.time(0, 0, 0)
        endtime = pd.to_datetime(data.get('endtime'), errors='coerce').time() if data.get('endtime') else datetime.time(23, 59, 59)
        
        print(district)
        
        
        a=pd.read_csv(r'home\datasets\FINAL_KARNATAKA_DATA.csv')
        districts_to_keep = [district]
        filtered_data = a[a['DISTRICTNAME'].isin(districts_to_keep)]
        filtered_data['Time'] = pd.to_datetime(filtered_data['Time'])
        
        filtered_data_time = filtered_data[filtered_data['Time'].dt.time.between(starttime, endtime)]
        filtered_data_time.to_csv(r'home\datasets\dataset.csv', index=False)
        print(len(filtered_data_time))



    #------------------------------------------------------------------------------------------------------------------------------
    #clustering
        from math import radians, sin, cos, sqrt, atan2

        # Load the dataset
        b = pd.read_csv(r'home\datasets\dataset.csv')

        # Define the distance thresholds (in kilometers)
        distance_threshold_1 = 0.3
        distance_threshold_2 = 1

        # Define the Haversine function to calculate the distance between two points
        def haversine(lat1, lon1, lat2, lon2):
            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = 6371 * c  # Radius of the Earth in kilometers
            return distance

        # Add a unique identifier to the original dataset
        b['Unique_ID'] = b.index

        # Group the dataset by 'DISTRICTNAME'
        grouped = b.groupby('DISTRICTNAME')

        region_data = []

        for district, group in grouped:
            region_mapping = {}
            for i, row in group.iterrows():
                lat, lon, road = row['Latitude'], row['Longitude'], row['Accident_Road']
                matched_region = None
                for region, (region_lat, region_lon, region_road) in region_mapping.items():
                    if haversine(lat, lon, region_lat, region_lon) <= distance_threshold_1:
                        matched_region = region
                        break
                    elif (distance_threshold_1 < haversine(lat, lon, region_lat, region_lon) <= distance_threshold_2 and
                        road == region_road):
                        matched_region = region
                        break
                if matched_region is None:
                    region_mapping[(lat, lon, road)] = (lat, lon, road)
                    matched_region = (lat, lon, road)
                region_data.append({'Latitude': lat, 'Longitude': lon, 'Accident_Road': road,
                                    'Region_Latitude': matched_region[0], 'Region_Longitude': matched_region[1], 'Region_Road': matched_region[2],
                                    'DISTRICTNAME': district, 'Unique_ID': row['Unique_ID']})

        # Create a DataFrame with region information
        region_df = pd.DataFrame(region_data)

        # Assign a region name to each unique region based on its latitude, longitude, and road
        region_df['Region_Name'] = region_df.groupby(['Region_Latitude', 'Region_Longitude', 'Region_Road']).ngroup().add(1)

        # Calculate total number of data points in each region
        region_counts = region_df.groupby(['Region_Name']).size().reset_index(name='Total_Data_Points')

        # Merge region counts with region dataframe
        region_df = pd.merge(region_df, region_counts, on='Region_Name')

        # Save the region information to a CSV file (optional)
        # Merge with the original dataset using the unique identifier to prevent duplication
        final_df = pd.merge(b, region_df, on=['Latitude', 'Longitude', 'DISTRICTNAME', 'Unique_ID'], how='left')

        # Sort the final dataframe by region name
        final_df.sort_values(by='Region_Name', inplace=True)
        final_df.to_csv(r'home\datasets\updated_dataset.csv', index=False)


        train_data_df = final_df.drop_duplicates(subset=['Region_Latitude', 'Region_Longitude'])
        train_data = train_data_df[['Total_Data_Points']]

        # Identify categorical and numerical columns
        numerical_cols = ['Total_Data_Points']

        # Define transformers
        numerical_transformer = StandardScaler()

        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
            ]
        )

        # Define the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', IsolationForest(contamination=0.1))
        ])

        pipeline.fit(train_data)
        anomalies = pipeline.predict(train_data)

        train_data_df['anomaly'] = anomalies
        train_data_df['anomaly'] = train_data_df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
        excluded_columns = ['Unique_ID', 'Crime_No', 'RI', 'Landmark_first', 
                        'landmark_second', 'Distance_LandMark_First', 
                        'Distance_LandMark_Second', 'Accident_Description']
    
        # Drop excluded columns
        train_data_df = train_data_df.drop(columns=excluded_columns, errors='ignore')
        

#         print("plotting starting")



  
# #-----------------------------------------------------------------------------------------------------------------
# #plotting      

#         # Create a map centered around Karnataka
#         from django.urls import reverse
#         first_row = train_data_df.iloc[1]
#         latitude = first_row['Region_Latitude']  # Replace with the exact column name
#         longitude = first_row['Region_Longitude']

#         # Assuming you have a view named 'analysis' in your Django app
#         analysis_url = reverse('telegram')  # Adjust this URL name to match your actual URL pattern


#         center_point = [latitude, longitude]  # Example center point for Mysuru City
#         m = folium.Map(location=center_point, zoom_start=10)
#         marker_cluster = MarkerCluster().add_to(m)
#         for index, row in train_data_df.iterrows():
#             if row['anomaly'] == 1:
#                 lat, lon = row['Region_Latitude'], row['Region_Longitude']
#                 popup_text = f'<a href="{analysis_url}?lat={lat}&lon={lon}" target="_blank">Click for analysis</a>'
#                 folium.CircleMarker(
#                             location=[lat, lon],
#                             popup=popup_text,  # Assigning the popup content to the marker
#                             radius=15,
#                             color='red',
#                             fill=True,
#                             fill_color='red'
#                 ).add_to(m)
#         map_html_path = r'static\assets\maps\mysore_map.html'
#         m.save(map_html_path)
#         print("done")
        anomaly_data = train_data_df[train_data_df['anomaly'] == 1]
        anomaly_data.fillna(value=0, inplace=True)


        # Convert the filtered DataFrame to a JSON-like format
        anomaly_json = anomaly_data.to_dict(orient='records')

        # Return the filtered data as a JSON response using Django REST Framework's Response
        return Response(anomaly_json)
        #response_data = {
        #'map_html_path': map_html_path,
        #}

        #return Response(response_data)

        



@login_required(login_url='/accounts/login/')
def alert(request):
  map_html_path = '' 
  if request.method == 'POST':
        place=request.POST.get('place')
        day = request.POST.get('day')
        start_date = pd.to_datetime(request.POST.get('startdate'))
        end_date= pd.to_datetime(request.POST.get('enddate'))
        starttime = pd.to_datetime(request.POST.get('starttime')).time()
        endtime = pd.to_datetime(request.POST.get('endtime')).time()
        print(start_date,end_date,day,starttime,endtime)
        df=pd.read_csv(r'home\datasets\traffic_data.csv')
        place=[place]
        df_place = df[df['traffic_police_station'].isin(place)]
        df_place['timestamp'] = pd.to_datetime(df_place['timestamp'])
        df_date = df_place[(df_place['timestamp'] >= start_date) & (df_place['timestamp'] <= end_date)]
        print(len(df_date))
        day = [day]
        df_day = df_date[df_date['day_of_week'].isin(day)]
        
        
        df_time = df_day[df_day['timestamp'].dt.time.between(starttime, endtime)]

        center_point = [17.45165, 78.3760]  # Example center point for Mysuru City
        m = folium.Map(location=center_point, zoom_start=14)
        marker_cluster = MarkerCluster().add_to(m)
        for index, row in df_time.iterrows():
            if row['anomaly'] == 1:
                lat, lon = row['latitude'], row['longitude']
                folium.CircleMarker(
                            location=[lat, lon],
                            radius=15,
                            color='red',
                            fill=True,
                            fill_color='red'
                ).add_to(m)
        map_html_path = r'static\assets\maps\traffic_congestion.html'
        m.save(map_html_path)

 
  return render(request, 'pages/alert.html', {'map_html_path': map_html_path})


@login_required(login_url='/accounts/login/')
def alert_explain(request):
    lat = float(request.GET.get('lat'))
    lon = float(request.GET.get('lon'))
    id="1lfqdrcUgjA7F0id3soULQULCmF9KAIglw1wSLisGywo"
    name="Accident"
    url="https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(id,name)
    df=pd.read_csv(url)
    rd = df[(df['Latitude'].astype(float) == lat) & (df['Longitude'].astype(float) == lon)]
    print(rd.iloc[0])
#---------------------------------------------------------------------------------------------------------------------
#"Latitude", "Longitude", "User Inconvinience Description", "Phone Number", "User Inconvinience Title", "Address", "Images"
    context = {
       'lat':rd["Latitude"].iloc[0],
       'lon': rd["Longitude"].iloc[0],
       'address': rd["Address"].iloc[0],
       'title': rd["User Inconvinience Title"].iloc[0],
       'desc': rd["User Inconvinience Description"].iloc[0],
       'ts': rd["Timestamp"].iloc[0],
       'img': rd["Img"].iloc[0],
       'pn': rd["Phone Number"].iloc[0],
       

    }

    return render(request, 'pages/alert_explain.html', {'context': context})



