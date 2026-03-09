"""
Streamlit demo application for AI Transparency Tools.

This module provides an interactive web interface for exploring XAI methods,
generating explanations, and evaluating explanation quality.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn

# Import our XAI tools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai_transparency_tools import ModelExplainer, DataLoader, DatasetMetadata
from ai_transparency_tools.eval import ExplanationEvaluator


# Page configuration
st.set_page_config(
    page_title="AI Transparency Tools",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 AI Transparency Tools</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This tool is for <strong>research and educational purposes only</strong>. 
    XAI outputs may be unstable or misleading and should not be used for 
    regulated decisions without human review. See the full disclaimer in the sidebar.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Dataset selection
    dataset_options = {
        "Iris": load_iris,
        "Wine": load_wine, 
        "Breast Cancer": load_breast_cancer
    }
    
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
    
    # Model selection
    model_options = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    # Explanation methods
    explanation_methods = st.multiselect(
        "Select Explanation Methods",
        ["SHAP", "LIME", "Integrated Gradients"],
        default=["SHAP", "LIME"]
    )
    
    # Evaluation parameters
    st.header("Evaluation Parameters")
    top_k_features = st.slider("Top-K Features for Evaluation", 1, 10, 5)
    n_samples_lime = st.slider("LIME Samples", 1000, 10000, 5000)
    
    # Disclaimer in sidebar
    st.header("Disclaimer")
    st.markdown("""
    This tool is designed for **research and educational purposes only**.
    
    **Limitations:**
    - XAI outputs may be unstable or misleading
    - Not a substitute for human judgment
    - Methodological limitations exist
    - Results may not generalize
    
    **Prohibited Use:**
    - DO NOT use for regulated decisions without human review
    - DO NOT rely solely on explanations for critical decisions
    - DO NOT use explanations to justify discriminatory decisions
    """)


def load_and_prepare_data(dataset_name):
    """Load and prepare dataset."""
    data_loader = DataLoader(random_state=42)
    
    if dataset_name == "Iris":
        X, y, metadata = data_loader.load_iris_dataset()
    elif dataset_name == "Wine":
        X, y, metadata = data_loader.load_wine_dataset()
    elif dataset_name == "Breast Cancer":
        X, y, metadata = data_loader.load_breast_cancer_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, metadata


def train_model(model, X_train, y_train):
    """Train the selected model."""
    model.fit(X_train, y_train)
    return model


def create_explanation_plots(explainer, X_test, method, instance_idx=0):
    """Create explanation visualizations."""
    if method == "SHAP":
        # SHAP summary plot
        explanation = explainer.explain_shap(X_test, explanation_type="local")
        shap_values = explanation["shap_values"]
        
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        # Create bar plot
        fig = px.bar(
            x=explainer.feature_names,
            y=np.mean(np.abs(shap_values), axis=0),
            title="SHAP Feature Importance (Global)",
            labels={"x": "Features", "y": "Mean |SHAP Value|"}
        )
        fig.update_layout(height=500)
        return fig
    
    elif method == "LIME":
        # LIME explanation
        explanations = explainer.explain_lime(X_test[:5], num_features=top_k_features)
        
        if explanations:
            explanation = explanations[instance_idx]
            features, scores = zip(*explanation["explanation"])
            
            # Create bar plot
            colors = ['red' if score < 0 else 'blue' for score in scores]
            fig = go.Figure(data=[
                go.Bar(x=list(features), y=list(scores), marker_color=colors)
            ])
            fig.update_layout(
                title=f"LIME Explanation for Instance {instance_idx}",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                height=500
            )
            return fig
    
    return None


def create_evaluation_dashboard(evaluation_results):
    """Create evaluation metrics dashboard."""
    st.header("📊 Explanation Quality Evaluation")
    
    # Create tabs for different evaluation aspects
    tab1, tab2, tab3, tab4 = st.tabs(["Faithfulness", "Stability", "Completeness", "Overall"])
    
    with tab1:
        st.subheader("Faithfulness Metrics")
        if "faithfulness" in evaluation_results:
            faithfulness = evaluation_results["faithfulness"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faithfulness Score", f"{faithfulness['faithfulness_score']:.3f}")
            with col2:
                st.metric("Standard Deviation", f"{faithfulness['faithfulness_std']:.3f}")
            with col3:
                st.metric("Method", faithfulness['method'])
    
    with tab2:
        st.subheader("Stability Metrics")
        if "stability" in evaluation_results:
            stability = evaluation_results["stability"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stability Score", f"{stability['stability_score']:.3f}")
            with col2:
                st.metric("Avg Correlation", f"{stability['average_rank_correlation']:.3f}")
            with col3:
                st.metric("Coefficient of Variation", f"{stability['coefficient_of_variation']:.3f}")
    
    with tab3:
        st.subheader("Completeness Metrics")
        if "completeness" in evaluation_results:
            completeness = evaluation_results["completeness"]
            
            # Create completeness plot
            top_k_values = []
            completeness_scores = []
            
            for key, value in completeness.items():
                if key.startswith("top_"):
                    top_k_values.append(int(key.split("_")[1]))
                    completeness_scores.append(value["mean_completeness"])
            
            fig = px.line(
                x=top_k_values, 
                y=completeness_scores,
                title="Completeness vs Top-K Features",
                labels={"x": "Top-K Features", "y": "Completeness Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Overall Evaluation")
        if "overall_score" in evaluation_results:
            overall = evaluation_results["overall_score"]
            
            # Create radar chart
            categories = ["Faithfulness", "Completeness", "Robustness"]
            values = [overall["faithfulness"], overall["completeness"], overall["robustness"]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Explanation Quality'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Overall Explanation Quality"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Weighted Overall Score", f"{overall['weighted_score']:.3f}")


def main():
    """Main application function."""
    
    # Load data
    with st.spinner("Loading dataset..."):
        X_train, X_test, y_train, y_test, metadata = load_and_prepare_data(selected_dataset)
    
    # Train model
    with st.spinner("Training model..."):
        trained_model = train_model(selected_model, X_train, y_train)
    
    # Calculate accuracy
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display model performance
    st.header("🎯 Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Samples", len(X_train))
    with col3:
        st.metric("Test Samples", len(X_test))
    
    # Initialize explainer
    with st.spinner("Initializing explainer..."):
        explainer = ModelExplainer(
            trained_model,
            X_train,
            y_train,
            metadata=metadata,
            random_state=42
        )
    
    # Display available methods
    available_methods = explainer.get_available_methods()
    st.info(f"Available explanation methods: {', '.join(available_methods)}")
    
    # Explanation generation
    st.header("🔍 Generate Explanations")
    
    # Method comparison
    if len(explanation_methods) > 1:
        st.subheader("Method Comparison")
        
        comparison_results = {}
        for method in explanation_methods:
            if method.lower() in available_methods:
                try:
                    if method == "SHAP":
                        result = explainer.explain_shap(X_test[:10], explanation_type="local")
                        comparison_results[method] = result
                    elif method == "LIME":
                        result = explainer.explain_lime(X_test[:5], num_features=top_k_features)
                        comparison_results[method] = result
                except Exception as e:
                    st.error(f"Error with {method}: {str(e)}")
        
        # Display comparison
        if comparison_results:
            st.subheader("Feature Importance Comparison")
            
            # Create comparison plot
            methods = list(comparison_results.keys())
            feature_names = explainer.feature_names
            
            fig = make_subplots(
                rows=len(methods), cols=1,
                subplot_titles=methods,
                vertical_spacing=0.1
            )
            
            for i, method in enumerate(methods):
                if method == "SHAP":
                    shap_values = comparison_results[method]["shap_values"]
                    if isinstance(shap_values, list):
                        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    importance = np.mean(np.abs(shap_values), axis=0)
                elif method == "LIME":
                    explanations = comparison_results[method]
                    importance = np.zeros(len(feature_names))
                    for explanation in explanations:
                        for feature, score in explanation["explanation"]:
                            if feature in feature_names:
                                idx = feature_names.index(feature)
                                importance[idx] += abs(score)
                    importance = importance / len(explanations)
                
                fig.add_trace(
                    go.Bar(x=feature_names, y=importance, name=method),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=300*len(methods), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Individual method analysis
    for method in explanation_methods:
        if method.lower() in available_methods:
            st.subheader(f"{method} Analysis")
            
            try:
                # Generate explanations
                if method == "SHAP":
                    explanation = explainer.explain_shap(X_test[:20], explanation_type="local")
                    
                    # Display SHAP summary
                    st.write("**SHAP Summary:**")
                    shap_values = explanation["shap_values"]
                    if isinstance(shap_values, list):
                        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    
                    # Feature importance table
                    importance_df = pd.DataFrame({
                        "Feature": explainer.feature_names,
                        "Mean |SHAP Value|": np.mean(np.abs(shap_values), axis=0)
                    }).sort_values("Mean |SHAP Value|", ascending=False)
                    
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # SHAP plot
                    fig = create_explanation_plots(explainer, X_test, method)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif method == "LIME":
                    explanations = explainer.explain_lime(X_test[:5], num_features=top_k_features)
                    
                    # Display LIME explanations
                    st.write("**LIME Explanations:**")
                    
                    for i, explanation in enumerate(explanations):
                        st.write(f"**Instance {i}:**")
                        explanation_df = pd.DataFrame(
                            explanation["explanation"],
                            columns=["Feature", "Importance"]
                        )
                        st.dataframe(explanation_df, use_container_width=True)
                        
                        # LIME plot
                        fig = create_explanation_plots(explainer, X_test, method, i)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error generating {method} explanations: {str(e)}")
    
    # Evaluation
    if st.button("🔬 Evaluate Explanation Quality"):
        with st.spinner("Evaluating explanation quality..."):
            evaluator = ExplanationEvaluator(random_state=42)
            
            # Get explanations for evaluation
            shap_explanation = explainer.explain_shap(X_test[:20], explanation_type="local")
            shap_values = shap_explanation["shap_values"]
            
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            # Generate evaluation report
            evaluation_results = evaluator.generate_evaluation_report(
                trained_model, X_test[:20], y_test[:20], shap_values, "SHAP"
            )
            
            # Display evaluation dashboard
            create_evaluation_dashboard(evaluation_results)
    
    # Dataset information
    with st.expander("📋 Dataset Information"):
        st.write("**Dataset Metadata:**")
        st.json(metadata.to_dict())
        
        # Feature statistics
        st.write("**Feature Statistics:**")
        feature_stats = pd.DataFrame(X_train, columns=explainer.feature_names).describe()
        st.dataframe(feature_stats, use_container_width=True)
    
    # Model information
    with st.expander("🤖 Model Information"):
        model_info = explainer.get_model_info()
        st.json(model_info)


if __name__ == "__main__":
    main()
