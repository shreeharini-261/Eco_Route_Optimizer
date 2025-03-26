import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_better_data():
    """Generate more realistic training data with patterns"""
    data = {
        'hour_of_day': np.concatenate([
            np.random.randint(7, 10, 300),  # Morning rush
            np.random.randint(16, 19, 300), # Evening rush
            np.random.randint(0, 24, 400)   # Other times
        ]),
        'day_of_week': np.concatenate([
            np.random.randint(0, 5, 700),   # Weekdays
            np.random.randint(5, 7, 300)    # Weekends
        ]),
        'start_lat': np.random.uniform(12.8, 13.2, 1000),
        'start_lng': np.random.uniform(80.0, 80.3, 1000),
        'end_lat': np.random.uniform(12.8, 13.2, 1000),
        'end_lng': np.random.uniform(80.0, 80.3, 1000),
        'traffic_level': np.concatenate([
            np.random.randint(3, 5, 600),   # High traffic during rush hours
            np.random.randint(1, 3, 400)    # Lower traffic otherwise
        ])
    }
    
    # Create realistic route preferences
    df = pd.DataFrame(data)
    df['optimal_route'] = np.where(
    (df['hour_of_day'].between(7, 9)) | (df['hour_of_day'].between(16, 18)),
    np.where(df['traffic_level'] > 3, 'route2', 'route1'),
    'route3'
)
    
    return df

def train_and_save_model():
    df = generate_better_data()
    
    # Feature engineering
    df['distance'] = np.sqrt((df['end_lat'] - df['start_lat'])**2 + 
                     (df['end_lng'] - df['start_lng'])**2)
    df['direction'] = np.arctan2(df['end_lat'] - df['start_lat'],
                                df['end_lng'] - df['start_lng'])
    
    # Encode routes
    le = LabelEncoder()
    df['route_encoded'] = le.fit_transform(df['optimal_route'])
    
    X = df[['hour_of_day', 'day_of_week', 'start_lat', 'start_lng', 
            'end_lat', 'end_lng', 'distance', 'traffic_level', 'direction']]
    y = df['route_encoded']
    
    # Train better model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    dump({'model': model, 'encoder': le}, 'models/traffic_model.pkl')
    print(f"Model trained with accuracy: {model.score(X_test, y_test):.2f}")
    print("Class distribution:", dict(zip(le.classes_, np.bincount(y))))

if __name__ == '__main__':
    train_and_save_model()