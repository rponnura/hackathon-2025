import pandas as pd
import numpy as np
import argparse
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import requests

# Suppress the DataFrame copy warning
pd.options.mode.chained_assignment = None  # default='warn'

def listwise_ranking(agents_df, intent, metrics_weights=None, performance_model=None):
    """
    Rank agents using a list-wise approach with LambdaMART-like scoring.
    
    Args:
        agents_df: DataFrame containing agent data
        intent: The customer intent
        metrics_weights: Dictionary of weights for different metrics
        performance_model: Optional trained model to predict agent performance
    
    Returns:
        DataFrame of ranked agents
    """
    # Create a deep copy to avoid the SettingWithCopyWarning
    df = agents_df.copy(deep=True)
    
    # Default weights if not provided
    if metrics_weights is None:
        metrics_weights = {
            'avg_sentiment_score': 0.4,  # Higher is better
            'avg_silence': -0.3,         # Lower is better (negative weight)
            'avg_acw': -0.3,             # Lower is better (negative weight)
            'skill_proficiency': 0.2,    # Higher is better
            'language_proficiency': 0.1  # Higher is better
        }
    
    # Convert intents to string to avoid the .str accessor error
    df['intents'] = df['intents'].astype(str).str.strip().str.lower()
    intent = str(intent).strip().lower()
    
    # Filter agents who have handled this intent
    matching_agents = df[df['intents'].str.contains(intent, case=False, na=False)]
    print(f"Found {len(matching_agents)} agents who have handled intent: '{intent}'")
    
    # If no exact matches, try partial matching
    if len(matching_agents) == 0:
        print(f"No exact matches for '{intent}'. Trying partial matching...")
        # Look for any intent that contains the search term
        matching_agents = df[df['intents'].apply(lambda x: intent in str(x).lower())]
        print(f"Found {len(matching_agents)} agents with partial intent matches")
        
        # If still no matches, return all agents
        if len(matching_agents) == 0:
            print(f"No matches found for '{intent}'. Showing all agents ranked by performance...")
            matching_agents = df
    
    # Create a deep copy of matching_agents to avoid the SettingWithCopyWarning
    active_agents = matching_agents.copy(deep=True)
    
    # Filter active agents
    if 'state' in active_agents.columns:
        # Check if 'state' is already numeric
        if active_agents['state'].dtype == 'object':
            # Convert 'active'/'inactive' to 1/0
            active_agents.loc[:, 'state_numeric'] = active_agents['state'].map({'active': 1, 'inactive': 0})
            active_agents = active_agents[active_agents['state_numeric'] == 1]
        else:
            # Assume 1 means active
            active_agents = active_agents[active_agents['state'] == 1]
        
        print(f"Found {len(active_agents)} active agents")
    else:
        print("Warning: 'state' column not found. Using all matching agents.")
    
    if len(active_agents) == 0:
        print("No active agents found. Using all matching agents instead.")
        active_agents = matching_agents.copy(deep=True)
    
    # If we have a trained model, use it to predict performance
    if performance_model is not None:
        try:
            # Prepare features for prediction
            features = prepare_features_for_prediction(active_agents, intent)
            
            # Predict performance scores
            predicted_scores = performance_model.predict(features)
            
            # Add predicted scores to the DataFrame
            active_agents.loc[:, 'predicted_performance'] = predicted_scores
            
            print(f"Added model predictions for {len(active_agents)} agents")
            
            # Add predicted performance to the metrics weights
            metrics_weights['predicted_performance'] = 0.5  # Give it a high weight
        except Exception as e:
            print(f"Error making predictions: {e}")
    
    # Calculate relevance scores for each metric
    for metric, weight in metrics_weights.items():
        if metric not in active_agents.columns:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            continue
            
        # Get min and max values
        min_val = active_agents[metric].min()
        max_val = active_agents[metric].max()
        range_val = max_val - min_val
        
        if range_val > 0:
            # Calculate relevance score (0-1)
            if weight > 0:  # Higher is better
                active_agents.loc[:, f'{metric}_rel'] = (active_agents[metric] - min_val) / range_val
            else:  # Lower is better
                active_agents.loc[:, f'{metric}_rel'] = 1 - ((active_agents[metric] - min_val) / range_val)
        else:
            active_agents.loc[:, f'{metric}_rel'] = 0.5  # Default if all values are the same
    
    # Calculate Discounted Cumulative Gain (DCG) for each agent
    active_agents.loc[:, 'dcg'] = 0
    
    for metric, weight in metrics_weights.items():
        if f'{metric}_rel' in active_agents.columns:
            # Use absolute weight value since direction is already accounted for in relevance
            active_agents.loc[:, 'dcg'] += abs(weight) * active_agents[f'{metric}_rel']
    
    # Sort by DCG (descending)
    ranked_agents = active_agents.sort_values('dcg', ascending=False)
    
    # Add rank position
    ranked_agents.loc[:, 'rank'] = range(1, len(ranked_agents) + 1)
    
    return ranked_agents

def prepare_data_for_training(df):
    """
    Prepare data for training the performance prediction model.
    
    Args:
        df: DataFrame containing agent data
    
    Returns:
        X: Features for training
        y: Target variable
        feature_names: Names of the features
    """
    print("Preparing data for training...")
    
    # Create a copy of the data
    data = df.copy(deep=True)
    
    # Clean up the data
    data.fillna(0, inplace=True)
    
    # Convert categorical variables
    if 'state' in data.columns and data['state'].dtype == 'object':
        data['state'] = data['state'].map({'active': 1, 'inactive': 0})
    
    if 'isOnCall' in data.columns and data['isOnCall'].dtype == 'object':
        data['isOnCall'] = data['isOnCall'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    
    # Create a performance score based on metrics
    # Higher sentiment is better, lower silence and ACW are better
    data['performance_score'] = (
        data['avg_sentiment_score'] * 0.4 - 
        data['avg_silence'] * 0.3 - 
        data['avg_acw'] * 0.3
    )
    
    # Encode intents
    intent_encoder = LabelEncoder()
    data['intent_encoded'] = intent_encoder.fit_transform(data['intents'].astype(str))
    
    # Save the encoder for later use
    with open('intent_encoder.pkl', 'wb') as f:
        pickle.dump(intent_encoder, f)
    
    # Select features for training
    feature_columns = [
        'intent_encoded', 'state', 'skill_proficiency', 'language_proficiency',
        'avg_silence', 'avg_acw', 'adherence_duration'
    ]
    
    # Only include columns that exist
    feature_columns = [col for col in feature_columns if col in data.columns]
    
    # Create feature matrix
    X = data[feature_columns]
    
    # Target variable: performance score
    y = data['performance_score']
    
    print(f"Prepared {len(X)} samples with {len(feature_columns)} features")
    print(f"Features: {feature_columns}")
    
    return X, y, feature_columns

def prepare_features_for_prediction(df, intent):
    """
    Prepare features for making predictions with the trained model.
    
    Args:
        df: DataFrame containing agent data
        intent: The customer intent
    
    Returns:
        X: Features for prediction
    """
    # Create a copy of the data
    data = df.copy(deep=True)
    
    # Load the intent encoder
    try:
        with open('intent_encoder.pkl', 'rb') as f:
            intent_encoder = pickle.load(f)
        
        # Encode the intent
        try:
            intent_encoded = intent_encoder.transform([intent.strip().lower()])[0]
        except:
            # If the intent is not in the encoder, use -1
            intent_encoded = -1
        
        # Add the encoded intent to the data
        data['intent_encoded'] = intent_encoded
    except:
        # If we can't load the encoder, use a default value
        data['intent_encoded'] = 0
    
    # Select the same features used for training
    feature_columns = [
        'intent_encoded', 'state', 'skill_proficiency', 'language_proficiency',
        'avg_silence', 'avg_acw', 'adherence_duration'
    ]
    
    # Only include columns that exist
    feature_columns = [col for col in feature_columns if col in data.columns]
    
    # Create feature matrix
    X = data[feature_columns]
    
    return X

def train_performance_model(df, model_path='agent_performance_model.pkl'):
    """
    Train a model to predict agent performance.
    
    Args:
        df: DataFrame containing agent data
        model_path: Path to save the trained model
    
    Returns:
        Trained model
    """
    print("Training agent performance prediction model...")
    
    # Prepare data for training
    X, y, feature_names = prepare_data_for_training(df)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model trained. Test RMSE: {rmse:.4f}")
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for i, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return model

def load_performance_model(model_path='agent_performance_model.pkl'):
    """
    Load a trained performance prediction model.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model or None if the model doesn't exist
    """
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded performance model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file {model_path} not found")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Agent recommendation based on intent")
    parser.add_argument('intent', nargs='?', default='', type=str, 
                        help='Customer intent (e.g., payment, balance, etc.)')
    parser.add_argument('--top', type=int, default=1, help='Number of top agents to recommend')
    parser.add_argument('--data', type=str, default="agent_new_metrics.csv", 
                        help='Path to agent metrics CSV file')
    parser.add_argument('--train', action='store_true', help='Train a new performance model')
    parser.add_argument('--no-model', action='store_true', help='Don\'t use the performance model')
    
    args = parser.parse_args()
    intent = args.intent
    top_n = args.top
    data_path = args.data

    if intent == "":
        data = {
            "Industry": "Banking",
            "Customer Action/Query": "what's up?",
            "Customer Type": "Standard", 
            "Customer Profile": "Engaged User"
        }
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        resp = response.json()
        intent = resp["Inferred Intent"]
        print("inferred intent is ", intent)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Clean up the data
    df.fillna(0, inplace=True)  # replaces all NaNs with 0
    
    # Train or load the performance model
    performance_model = None
    if not args.no_model:
        if args.train:
            performance_model = train_performance_model(df)
        else:
            performance_model = load_performance_model()
    
    print(f"Finding top {top_n} agents for intent: '{intent}'...")
    
    # Get top agents using listwise ranking
    top_agents = listwise_ranking(df, intent, performance_model=performance_model)
    
    if top_agents.empty:
        print(f"No agents found for intent: '{intent}'")
        return
    
    # Display top N agents
    print(f"\nTop {top_n} agents for intent '{intent}':")
    top_n_agents = top_agents.head(top_n)
    
    # Select relevant columns for display
    display_columns = ['user id', 'user name', 'division name', 'dcg', 'rank', 
                      'avg_sentiment_score', 'avg_silence', 'avg_acw']
    
    # Add predicted performance if available
    if 'predicted_performance' in top_n_agents.columns:
        display_columns.append('predicted_performance')
    
    # Only include columns that exist in the DataFrame
    existing_columns = [col for col in display_columns if col in top_n_agents.columns]
    
    print(top_n_agents[existing_columns].to_string(index=False))
    
    # Print the top agent in a clear format
    if not top_n_agents.empty:
        top_agent = top_n_agents.iloc[0]
        print("\n" + "="*50)
        print(f"TOP AGENT FOR INTENT '{intent.upper()}':")
        print("="*50)
        
        if 'user name' in top_agent:
            print(f"Name: {top_agent['user name']}")
        
        if 'division name' in top_agent:
            print(f"Division: {top_agent['division name']}")
        
        if 'avg_sentiment_score' in top_agent:
            print(f"Sentiment Score: {top_agent['avg_sentiment_score']:.4f}")
        
        if 'avg_silence' in top_agent:
            print(f"Average Silence: {top_agent['avg_silence']:.2f} seconds")
        
        if 'avg_acw' in top_agent:
            print(f"Average ACW: {top_agent['avg_acw']:.4f}")
        
        if 'predicted_performance' in top_agent:
            print(f"Predicted Performance: {top_agent['predicted_performance']:.4f}")
        
        if 'dcg' in top_agent:
            print(f"Overall Score: {top_agent['dcg']:.4f}")
        
        print("="*50)

if __name__ == "__main__":
    main()