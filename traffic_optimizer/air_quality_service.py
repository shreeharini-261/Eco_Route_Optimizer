import requests
import os

GOOGLE_CLOUD_API_KEY = 'AIzaSyAdmglUyvS-6HhObW_oYyF1aolSE6M6GyM'  # Replace with your actual key

def get_air_quality(latitude, longitude):
    """
    Get air quality data from Google Cloud Air Quality API
    Returns a dictionary with air quality information
    """
    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={GOOGLE_CLOUD_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "extraComputations": [
            "HEALTH_RECOMMENDATIONS",
            "DOMINANT_POLLUTANT_CONCENTRATION",
            "POLLUTANT_CONCENTRATION"
        ],
        "languageCode": "en"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant information
        aqi = data.get('indexes', [{}])[0].get('aqi', 'N/A')
        dominant_pollutant = data.get('indexes', [{}])[0].get('dominantPollutant', 'N/A')
        category = data.get('indexes', [{}])[0].get('category', 'N/A')
        health_recommendations = data.get('healthRecommendations', {}).get('generalPopulation', 'N/A')
        
        return {
            'aqi': aqi,
            'dominant_pollutant': dominant_pollutant,
            'category': category,
            'health_recommendations': health_recommendations,
            'success': True
        }
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        return {
            'success': False,
            'error': str(e)
        }