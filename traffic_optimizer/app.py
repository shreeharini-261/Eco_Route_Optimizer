from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import requests
import numpy as np
from joblib import load
from datetime import datetime
import pandas as pd
import os
from air_quality_service import get_air_quality
import subprocess
import json
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load trained model
try:
    model_data = load('models/traffic_model.pkl')
    model = model_data['model']
    le = model_data['encoder']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    le = None

# API Keys
GOOGLE_MAPS_API_KEY = 'AIzaSyAdmglUyvS-6HhObW_oYyF1aolSE6M6GyM'

def get_coordinates(location):
    """Convert address to lat/lng using Google Geocoding API"""
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(geocode_url)
    data = response.json()
    if data.get('results'):
        return data['results'][0]['geometry']['location']
    return None

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                      (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', api_key=GOOGLE_MAPS_API_KEY)

# Update the optimized_route endpoint
@app.route('/optimized_route', methods=['POST'])
def get_optimized_route():
    try:
        data = request.get_json()
        origin = data.get('origin', '').strip()
        destination = data.get('destination', '').strip()
        
        if not origin or not destination:
            return jsonify({'status': 'error', 'error': 'Both origin and destination are required'}), 400

        # Validate addresses don't contain problematic characters
        if any(char in origin + destination for char in ['<', '>', '"', '{', '}', '|', '\\', '^', '~', '[', ']', '`']):
            return jsonify({'status': 'error', 'error': 'Address contains invalid characters'}), 400

        # Get coordinates for air quality midpoint calculation
        origin_coords = get_coordinates(origin)
        dest_coords = get_coordinates(destination)
        
        if not origin_coords or not dest_coords:
            return jsonify({'status': 'error', 'error': 'Could not geocode locations'}), 400

        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs,routes.travelAdvisory'
        }
        
        payload = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "DRIVE",
            "computeAlternativeRoutes": True,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": True
            },
            "languageCode": "en-US",
            "units": "METRIC"
        }

        # Make API request with timeout
        response = requests.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown API error')
            return jsonify({'status': 'error', 'error': f'Google API error: {error_msg}'}), 400

        data = response.json()

        if not data.get('routes'):
            return jsonify({'status': 'error', 'error': 'No routes found between locations'}), 404

        # Process all routes
        evaluated_routes = []
        for i, route in enumerate(data.get('routes', [])):
            # Calculate midpoint for air quality check
            midpoint = {
                'lat': (origin_coords['lat'] + dest_coords['lat']) / 2,
                'lng': (origin_coords['lng'] + dest_coords['lng']) / 2
            }
            
            # Get air quality data
            aq_data = get_air_quality(midpoint['lat'], midpoint['lng'])
            
            # Extract route metrics
            duration = float(route.get('duration', '0s').replace('s', ''))
            distance = route.get('distanceMeters', 0)
            aqi = int(aq_data.get('aqi', 100)) if aq_data.get('success') else 100
            
            # Normalize values for scoring
            norm_duration = duration / 3600  # seconds to hours
            norm_distance = distance / 1000  # meters to km
            norm_aqi = aqi / 500  # scale AQI
            
            # Calculate score (lower is better)
            score = 0.6 * norm_duration + 0.2 * norm_distance + 0.2 * norm_aqi
            
            evaluated_routes.append({
                'route': route,
                'air_quality': aq_data,
                'score': score,
                'route_index': i,
                'origin': origin,
                'destination': destination,
                'duration_seconds': duration,
                'distance_meters': distance
            })

        # Sort routes by score (best first)
        evaluated_routes.sort(key=lambda x: x['score'])
        best_route = evaluated_routes[0]
        
        # Prepare alternative routes data
        alternative_routes = []
        for i, route in enumerate(evaluated_routes[1:]):  # Skip best route
            alternative_routes.append({
                'polyline': route['route']['polyline']['encodedPolyline'],
                'distance_meters': route['distance_meters'],
                'duration_seconds': route['duration_seconds'],
                'score': route['score']
            })

        return jsonify({
            'status': 'success',
            'polyline': best_route['route']['polyline']['encodedPolyline'],
            'distance_meters': best_route['distance_meters'],
            'duration_seconds': best_route['duration_seconds'],
            'air_quality': best_route['air_quality'],
            'score': best_route['score'],
            'origin': best_route['origin'],
            'destination': best_route['destination'],
            'legs': best_route['route'].get('legs', []),
            'alternative_routes': alternative_routes,
            'travel_advisory': best_route['route'].get('travelAdvisory', {})
        })

    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'error': 'Request timed out. Please try again.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f'Unexpected error in optimized_route: {str(e)}')
        return jsonify({'status': 'error', 'error': 'An unexpected error occurred'}), 500
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        route_data = data.get('route_data', {})
        air_quality_data = data.get('air_quality_data', {})
        
        context = {
            "user_message": user_message,
            "route_data": route_data,
            "air_quality_data": air_quality_data,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result = query_llama(context)
        return jsonify({'response': result})
        
    except Exception as e:
        print(f"Error in AI endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def query_llama(context):
    """Query the Llama 3.2 model with the given context"""
    try:
        prompt = f"""
        You are an AI assistant for a multi-modal transportation system. 
        User question: "{context['user_message']}"
        
        Context:
        - Route Data: {json.dumps(context.get('route_data', {}), indent=2)}
        - Air Quality: {json.dumps(context.get('air_quality_data', {}), indent=2)}
        
        Provide recommendations considering:
        - Travel time
        - Air quality impact
        - Health benefits
        - Available transport modes (drive, transit, bike, walk)
        """
        
        cmd = f"ollama run llama3.2 '{prompt}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Llama query failed: {result.stderr}")
            
        return result.stdout.strip()
        
    except Exception as e:
        print(f"Error querying Llama: {e}")
        return "I'm having trouble processing your request. Please try again later."

if __name__ == '__main__':
    app.run(debug=True, port=5000)