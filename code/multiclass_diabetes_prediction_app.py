
# Import the neccesary libraries

# Web app
import streamlit as st

# Tabular and matricial data handling
import pandas as pd
import numpy as np

# Serialization
from joblib import load

# Preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Performance metrics
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Figures
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib

# Feature importance
import shap
from sklearn.inspection import permutation_importance



def display_centered_df(df):
    '''
    Fuction to display a dataframe in the center
    '''
    # Create 3 columns, being the one in the middle the wider
    column_1, column_2, column_3 = st.columns([1, 34, 1])
    
    # Display the dataframe in the middle
    with column_2:
        st.dataframe(df)


def evaluate_model(model, X_test, y_test, encoder):
    '''
    Function to evaluate the model with confusion matrix, accuracy, and macro average precision, recall and F1
    '''

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Get the original labels from the encoder
    display_labels = encoder.inverse_transform(model.classes_)

    # Convert labels to list
    display_labels = list(display_labels)

    # Reverse the y-axis (so that the confusion matrix is plotted as usual)
    reversed_labels = display_labels[::-1]

    # Modify the text that is overlayed with the user mouse
    text_to_overlay = [
        [f'True: {y_label}<br>Predicted: {x_label}<br>Num. cases: {value}' for x_label, value in zip(display_labels, row)]
        for y_label, row in zip(reversed_labels, conf_matrix[::-1])
    ]

    # Create a confusion matrix with a heatmap
    fig = ff.create_annotated_heatmap(
        z = conf_matrix[::-1],
        x = display_labels,
        y = reversed_labels,
        colorscale = 'Blues',
        hoverinfo = 'text',
        text = text_to_overlay
    )

    # Set the X- and Y-axis labels, and the size of the plot
    fig.update_layout(
        title = {
            'text': '',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title = 'Predicted Label', 
        yaxis_title = 'True Label',
        width = 600,
        height = 600,
        xaxis = dict(side = 'bottom')
    )

    # Set the subheader for the confusion matrix
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Confusion Matrix")

    # Show the figure
    st.plotly_chart(fig)

    # Compute accuracy, and macro average precision, recall and F1
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # Create a dataframe with the performance metrics
    metrics = {
        'Accuracy': accuracy, 
        'Macro-Average F1 Score': f1_macro, 
        'Macro-Average Precision': precision, 
        'Macro-Average Recall': recall
    }
    metrics_dataframe = pd.DataFrame([metrics])
    
    # Set the index
    metrics_dataframe.index = ['Performance value']
    metrics_dataframe.index.name = "Performance metric"
    
    # Set the subheader for the performance metrics
    st.subheader("Predictive Performance Metrics")
    st.markdown("<br>", unsafe_allow_html=True)

    # Show the dataframe
    display_centered_df(metrics_dataframe)


def plot_feature_importances(pipeline, feature_names):
    """
    Function to plot the feature importance of a model in a pipeline.
    """
    
    # Get the model
    model = pipeline.named_steps['model']

    # Get only the feature names that were selected in feature selection
    feeature_selector = pipeline.named_steps['feature_selection']
    feeature_selector_mask = feeature_selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if feeature_selector_mask[i]]

    # Get the feature importances
    feature_importances = model.feature_importances_

    # Create a dataframe for the lpot
    feature_importances_dataframe = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances
    })

    # Sorting the features by their importance
    feature_importances_dataframe = feature_importances_dataframe.sort_values(by = 'Importance', ascending = True)

    # Get colors from the tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_importances_dataframe)))

    # Plot the feature importances with a bar plot
    # The X-axis contains the importance values and the y-axis the feature names
    plt.figure(figsize = (10, 6))
    plt.barh(feature_importances_dataframe['Feature'], feature_importances_dataframe['Importance'], color = colors)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    st.pyplot(plt)


def plot_permutation_importances(pipeline, X_test, y_test, feature_names):
    """
    Function to plot the mean permutation importances of a model in a pipeline. The performance metric is macro-average F1 score.
    The feature importances are expressed as percentages and error bars are also plotted with the standard deviation.
    """
    
    # Set the subheader for permutation feature
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Permutation Feature Importance")

    # Apply the scaling to the testing data (same transformation as training)
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)

    # Select the features that were selected in the feature selection
    feature_selector = pipeline.named_steps['feature_selection']
    feature_selector_mask = feature_selector.get_support()
    X_test_scaled_selected = X_test_scaled[:,feature_selector_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if feature_selector_mask[i]]

    # Get the model
    model = pipeline.named_steps['model']

    # Compute permutation importance using the macro average F1 score as the performance metric and 50 repeats
    permutation_importance_results = permutation_importance(model, X_test_scaled_selected, y_test, scoring = 'f1_macro', n_repeats = 50, random_state = 42, n_jobs = -1)

    # Sort the results by the importance value and convert the results to percentages for easier interpretation to the user
    permutation_importance_results_sorted_indices = permutation_importance_results.importances_mean.argsort()
    sorted_mean_permutation_importance = permutation_importance_results.importances_mean[permutation_importance_results_sorted_indices] * 100
    sorted_sd_permutation_importance = permutation_importance_results.importances_std[permutation_importance_results_sorted_indices] * 100
    sorted_features = [feature_names[i] for i in permutation_importance_results_sorted_indices]

    # Get colors in the RGBA format from the tab10 colormap
    tab10_colors = [f'rgba({r*255}, {g*255}, {b*255}, {a})' for r, g, b, a in plt.cm.tab10(np.linspace(0, 1, len(sorted_features)))]

    # Plot the importances with a bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y = sorted_features,
        x = sorted_mean_permutation_importance,
        error_x = dict(type = 'data', array = adjusted_std, visible = True),    # Show the standard deviation with error bars
        orientation = 'h',
        marker_color = tab10_colors
    ))

    # Set and center the title, set the figure size, and set the X- and Y-axis labels
    fig.update_layout(
        title = {
            'text': 'Feature Importances by Permutation',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title = 'Permutation Importance (Mean Decrease in Macro-average F1 Score) [%]',
        yaxis_title = 'Feature',
        autosize = False,
        width = 800,
        height = 600
    )

    # For better readibility, increase the maximum value of the X-axis
    fig.update_xaxes(range=[0, max(sorted_mean_permutation_importance + adjusted_std) + 10])

    # Include the importance values with text annotation at the right of the bars
    annotations = []
    for yd, xd, std in zip(sorted_features, sorted_mean_permutation_importance, adjusted_std):
        annotations.append(dict(
            x = xd + std + 0.5,
            y = yd,
            text = f"{xd:.2f}%",
            showarrow = False,
            font = dict(size = 12),
            xanchor = 'left',
            yanchor = 'middle'
        ))
    fig.update_layout(annotations = annotations)

    # Show the plot
    st.plotly_chart(fig)

    return permutation_importance_results.importances_mean, feature_names


def plot_shap_importances(pipeline, X_test, feature_names, encoder):
    """
    Function to plot SHAP feature important figures for a model in a pipeline
    """

    # The figures in SHAP need JavaScript
    shap.initjs()

    # Apply the scaling to the testing data (same transformation as training)
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)

    # Select the features that were selected in the feature selection
    feature_selector = pipeline.named_steps['feature_selection']
    feature_selector_mask = feature_selector.get_support()
    X_test_scaled_selected = X_test_scaled[:,feature_selector_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if feature_selector_mask[i]]

    # Get the model
    model = pipeline.named_steps['model']

    # Compute the Shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled_selected)

    # Change the Shap values format to the format expected by the multiclass plot
    if isinstance(shap_values, np.ndarray):
        shap_values_multiclass = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    # Create colors with the tab10 colormap
    tab10_cmap = matplotlib.cm.get_cmap('tab10')
    tab10_colors = tab10_cmap(np.linspace(0, 1, 10))

    # Get the class named from the encoder
    class_names = encoder.inverse_transform(model.classes_)

    # Set the subheader for the Shap values separated by class
    st.markdown("<br>", unsafe_allow_html = True)
    st.subheader('SHAP Feature Importance by Class with Direction of the Effect')
    st.markdown("<br>", unsafe_allow_html = True)

    # Plot the Shap values separated by class with a beeswarm plot, which also shows the direction of the effect (not only the absolute value)
    for i in range(shap_values.shape[2]):
        figure = plt.figure(figsize = (10, 6))
        shap_values_current_class = shap_values[:, :, i]
        shap.summary_plot(shap_values_current_class, X_test_scaled_selected, feature_names = feature_names, plot_type = "dot", color = tab10_colors, show = False)
        plt.title(f"SHAP Feature Importance of Class {class_names[i]}")
        st.pyplot(figure)
        st.markdown("<br>", unsafe_allow_html = True)
        plt.clf()

    # Set the subheader
    st.subheader('Absolute Shap feature importance')

    # Show the magnitude (absolute value) of the average of the Shap values by feature in one plot, indicating the contribution of the feature to each class
    figure = plt.figure(figsize = (10, 6))
    plt.suptitle("Absolute Shap Feature Importance, Split by Class")
    shap.summary_plot(shap_values_multiclass, X_test_scaled_selected, feature_names = feature_names,
                      class_names = class_names, plot_type = "bar", color = tab10_cmap, show = False)
    st.pyplot(figure)
    plt.clf()

    # Compute the global feature importance, by averaging the Shap values both by class, and by feature
    magnitude_shap_values_mean_classes = np.mean(np.abs(shap_values), axis=2) # Average by class
    magnitude_shap_values_mean_features = np.mean(magnitude_shap_values_mean_classes, axis=0) # Average by feature

    return magnitude_shap_values_mean_features


def compute_plot_combined_importance(shap_importances, permutation_importances, feature_names):
    '''
    Function to compute amd plot the combined performance by averaging (after normalizing) the importances of permutation improtance and Shap
    '''

    # Set the subheader for the combined feature importance
    st.markdown("<br><br>", unsafe_allow_html = True)
    st.subheader("Combined Feature Importance")

    # Convert importances to numpy arrays
    shap_importances = np.array(shap_importances)
    permutation_importances = np.array(permutation_importances)

    # Normalize importance to be between 0 and 100, since they are in different scales
    shap_values_scaled = 100 * (shap_importances - np.min(shap_importances)) / (np.max(shap_importances) - np.min(shap_importances))
    permutation_values_scaled = 100 * (permutation_importances - np.min(permutation_importances)) / (np.max(permutation_importances) - np.min(permutation_importances))

    # Calculate the mean of normalized importances
    mean_importances = (shap_values_scaled + permutation_values_scaled) / 2

    # Sort by their importance value
    sorted_importances_indices = np.argsort(mean_importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_importances_indices]
    sorted_importances = mean_importances[sorted_importances_indices]

    # Get colors in the RGBA format from the tab10 colormap
    cmap = plt.cm.get_cmap('tab10')
    colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})' for r, g, b, a in cmap(np.linspace(0, 1, len(sorted_features)))]

    # Create a bar plot to show the feature importances
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y = sorted_features,
        x = sorted_importances,
        orientation = 'h',
        marker = dict(color = colors)
    ))

    # Set and center the title, set the X- and Y-axis labels, reverse the y-axis (from higher to lower) and set the figure size
    fig.update_layout(
        title = {
            'text': 'Average Feature Importance from SHAP and Permutation',
            'y': 0.901,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title = 'Average Normalized Importance from SHAP and Permutation',
        yaxis_title = 'Features',
        yaxis = dict(autorange = 'reversed'),
        margin = dict(l = 100, r = 20, t = 100, b = 70),
        height = 600,
        width = 800,
        template = 'plotly_white'
    )

    # Show the plot
    st.plotly_chart(fig)


def plot_importances_by_feature_type(shap_importances, permutation_importances, feature_names):
    '''
    Function to plot the Shap values grouped by the feature types.
    The feature types considered are:
        - Cholesterol / Lipids
        - Demographic
        - Physical Measurements
        - Renal
    '''

    # Set the subheader for the combined importance by feature type
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Combined Feature Type Importance")

    # Define the correspondence between feature types and features
    feature_types_to_feature_list = {
        'Cholesterol / Lipids': ['Chol', 'TG', 'LDL', 'TG/HDL', 'Non-HDL', 'VLDL'],
        'Demographic': ['Age', 'Sex'],
        'Physical Measurements': ['BMI'],
        'Renal': ['Cr']
    }

    # Create a mapping from features to feature types (the other way around)
    feature_to_feature_type = {}
    for feature_type, features in feature_types_to_feature_list.items():
        for feature in features:
            feature_to_feature_type[feature] = feature_type

    # Conver importances to numpy arrays
    shap_importances = np.array(shap_importances)
    permutation_importances = np.array(permutation_importances)

    # Normalize importance to be between 0 and 100, since they are in different scales
    shap_values_scaled = 100 * (shap_importances - np.min(shap_importances)) / (np.max(shap_importances) - np.min(shap_importances))
    permutation_values_scaled = 100 * (permutation_importances - np.min(permutation_importances)) / (np.max(permutation_importances) - np.min(permutation_importances))

    # Calculate the mean of normalized importances
    mean_importances = (shap_values_scaled + permutation_values_scaled) / 2

    # Append the importances of each feature type and sum it
    feature_type_importances = {}
    for feature, importance in zip(feature_names, mean_importances):
        feature_type = feature_to_feature_type.get(feature, 'Other')
        if feature_type not in feature_type_importances:
            feature_type_importances[feature_type] = []
        feature_type_importances[feature_type].append(importance)
    sum_feature_type_importances = {feature_type: np.sum(importances_values) for feature_type, importances_values in feature_type_importances.items()}

    # Sort the results by their importance value
    sorted_feature_types = sorted(sum_feature_type_importances, key = sum_feature_type_importances.get, reverse = True)
    sorted_importances = [sum_feature_type_importances[feature_type] for feature_type in sorted_feature_types]

    # Define the colorsin RGBA format
    colors = ['rgba(31, 119, 180, 1)',
              'rgba(255, 127, 14, 1)', 
              'rgba(44, 160, 44, 1)']

    # Create a bar plot to show the feature importances
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y = sorted_feature_types,
        x = sorted_importances,
        orientation = 'h',
        marker = dict(color = colors)
    ))

    # Set and center the title, set the X- and Y-axis labels, reverse the y-axis (from higher to lower) and set the figure size
    fig.update_layout(
        title = {
            'text': 'Average Feature Type Importance from SHAP and Permutation by Feature Type',
            'y': 0.901,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title = 'Average Normalized Importance from SHAP and Permutation',
        yaxis_title = 'Feature Type',
        yaxis = dict(autorange = 'reversed'),
        margin = dict(l = 100, r = 20, t = 100, b = 70),
        height = 400,
        width = 800,
        template = 'plotly_white'
    )

    # Show the plot
    st.plotly_chart(fig)


def plot_categorical_variable_distribution(categorical_vars, data):
    '''
    Function to plot the distribution of the categorical variables
    '''

    # Since there are only 2 categorical variables, create a figure with two subplots
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = [f"{variable} distribution" for variable in categorical_vars])

    # Get tab10 colors
    tab10_colors = px.colors.qualitative.T10

    # For each categorical variable
    for i, variable in enumerate(categorical_vars):

        # Count the frequency of each category
        counts = data[variable].value_counts().reset_index()
        counts.columns = [variable, 'Count']

        # Create a bar plot with the frequency distribution of the categories
        bar = go.Bar(
            x = counts[variable],
            y = counts['Count'],
            text = counts['Count'],
            textposition = 'outside',
            marker_color = tab10_colors[:len(counts)],
            name = variable
        ) 

        # Add the plot to the figure
        fig.add_trace(bar, row = 1, col = i+1)

    # Set the figure size
    fig.update_layout(
        height = 420,
        width = 700,
        showlegend = False,
        title = {
            'text': '',
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    # Do not rotate the X-axis labels
    fig.update_xaxes(tickangle = 0, tickmode = 'array')

    # In the Y-axis, add margin above the highest bar
    for i in range(1, len(categorical_vars) + 1):
        max_y_value = data[categorical_vars[i - 1]].value_counts().max()
        fig.update_yaxes(range = [0, max_y_value * 1.15], row = 1, col = i)

    # Show the figure
    st.plotly_chart(fig)


def plot_distribution_numerical_variables(numerical_variables, data):
    '''
    Function to plot the distribution of the numerical variables
    '''

    # Calculate the number of rows
    num_rows = (len(numerical_variables) + 1) // 2

    # If the number of variables is not even, add some offset
    height_offset, vertical_spacing_offet = 0, 0
    if len(numerical_variables) % 2 != 0:
        vertical_spacing_offet = 0.04
        height_offset = 50

    # Create subplots with the the calculated number of rows and 2 columns
    fig = make_subplots(rows = num_rows, cols = 2, subplot_titles = numerical_variables, vertical_spacing = 0.06+vertical_spacing_offet, horizontal_spacing = 0.2)

    # Get colors in the RGBA format from the tab10 colormap
    tab10_colors = [f'rgba({r*255}, {g*255}, {b*255}, {a})' for r, g, b, a in plt.cm.tab10(np.linspace(0, 1, len(numerical_variables)))]

    # For each variable
    for i, variable in enumerate(numerical_variables):

        # Calculate the current row cand column
        row = i // 2 + 1
        col = i % 2 + 1

        # Get the color
        color = tab10_colors[i % 10]
        
        # Create an histogram
        hist = go.Histogram(
            x = data[variable],
            nbinsx = 50,
            name = f"{variable} Distribution",
            marker = dict(color = color)
        )
        fig.add_trace(hist, row=row, col=col)
    
    # Set the figure size
    fig.update_layout(
        height = 300 * num_rows + height_offset,
        width = 700,
        showlegend = False,
        title = {
            'text': '',
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    # Show the plot
    st.plotly_chart(fig)


def perform_data_preprocessing(uploaded_file):
    '''
    Function to perform the data pre-processing, including the data loading, feature engineering, encoding and data splitting.
    '''

    # Load the data
    data = pd.read_csv(uploaded_file)
        
    # Split variables names by their type (categorical or numerical)
    categorical_vars = [variable for variable in data.columns if (data[variable].dtype == 'object')]
    numerical_vars = [variable for variable in data.columns if data[variable].dtype in ['int64', 'float64']]

    # Feature engineering
    data['TG/HDL'] = data['TG'] / data['HDL']       # TG / HDL
    data['Chol/HDL'] = data['Chol'] / data['HDL']   # Chol / HDL
    data['Non-HDL'] = data['Chol'] - data['HDL']    # Non-HDL (Total - HDL)
    data['LDL/HDL'] = data['LDL'] / data['HDL']     # LDL / HDL
    data['Urea/Cr'] = data['Urea'] / data['Cr']     # Urea / Cr

    #data_pre_encoding = data.drop('Num_Patient', axis=1).copy()
    data_pre_encoding = data.copy()

    # Create a separate encoder for each variable
    sex_encoder = LabelEncoder()
    diagnosis_encoder = LabelEncoder()

    # Encode the categorical variables
    data['Sex'] = sex_encoder.fit_transform(data['Sex'])
    data['Diagnosis'] = diagnosis_encoder.fit_transform(data['Diagnosis'])

    # Split the data into features and target
    X_test = data.drop('Diagnosis', axis=1)
    y_test = data['Diagnosis']

    # Load the trained model
    pipeline = load('data/best_model_pipeline.joblib')

    # Store in session state
    st.session_state.uploaded_file = uploaded_file
    st.session_state.data_no_glycemic = data
    st.session_state.categorical_vars = categorical_vars
    st.session_state.numerical_vars = numerical_vars
    st.session_state.data_pre_encoding = data_pre_encoding
    st.session_state.pipeline = pipeline
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.diagnosis_encoder = diagnosis_encoder


def perform_exploratory_data_analysis():
    '''
    Function to perform the exploratory data analysis, including the computation of the summary statistics, 
    and the plots with the distribution of the categorical and numerical variables
    '''

    # Show the summary statistics
    st.markdown("<br><br>", unsafe_allow_html = True)
    st.subheader('Summary Statistics')
    display_centered_df(st.session_state.data_pre_encoding.describe())

    # Show the distribution of the categorical variables with bar plots
    st.markdown("<br><br>", unsafe_allow_html = True)
    st.subheader('Distribution of Categorical Variables')
    plot_categorical_variable_distribution(st.session_state.categorical_vars, st.session_state.data_pre_encoding)

    # Show the distribution of the numerical variables with histograms
    st.subheader('Distribution of Available Numerical Variables')
    plot_distribution_numerical_variables(st.session_state.numerical_vars, st.session_state.data_pre_encoding)
    st.subheader('Distribution of Created Variables')
    created_variables = ['TG/HDL', 'Chol/HDL', 'Chol/HDL', 'LDL/HDL', 'Urea/Cr']
    plot_distribution_numerical_variables(created_variables, st.session_state.data_pre_encoding)


def perform_model_inspection():
    '''
    Function to perform the model inspection, including the computation and plots of permutation feature importance,
    the Shap values, the combined feature importance of both, and the importance by feature type
    '''

    # Plot model intrisic feature importance
    #plot_feature_importances(pipeline, X.columns)
    
    # Plot the permutation feature importance
    permutation_importance_values, feature_names = plot_permutation_importances(st.session_state.pipeline, st.session_state.X_test, st.session_state.y_test, list(st.session_state.X_test.columns))

    # Plot the SHAP feature importance
    shap_values = plot_shap_importances(st.session_state.pipeline, st.session_state.X_test, st.session_state.X_test.columns, st.session_state.diagnosis_encoder)

    # Compute and plot the combined feature importance of permutation importance and Shap
    compute_plot_combined_importance(shap_values, permutation_importance_values, feature_names)

    # Plot the combined importances grouped by feature type
    plot_importances_by_feature_type(shap_values, permutation_importance_values, feature_names)


def initialize_state_variables():
    '''
    Function to set the default values for the variables in the session
    '''
    variable_names = ["uploaded_file", "data_no_glycemic", "categorical_vars", "numerical_vars",
                      "data_pre_encoding", "pipeline", "X_test", "y_test", "diagnosis_encoder"]
    for variable in variable_names:
        if variable not in st.session_state:
            st.session_state[variable] = None


def config_page_settings():
    st.set_page_config(
        page_title='Multiclass Diabetes Prediction and Inspection',
        layout='centered',
        initial_sidebar_state='expanded'
    )


def show_introduction_explanation():
    st.markdown("""
        This web application classifies patients into three categories based on their biomarkers: Diabetes, Prediabetes, and Non-diabetes.

        ### App Tabs Overview

        #### Introduction
        This tab provides an overview of the application's operation, details the predictive model used, and the performance metrics achieved. It also outlines the required dataset format. To facilitate testing and result reproduction, a sample dataset can be accessed [here](https://github.com/colme99/multiclass_diabetes_prediction/blob/main/data/test_data.csv).

        #### Dataset Upload
        The default tab where users upload their dataset in CSV format. Uploading data is necessary to enable functionality in other tabs.

        #### Exploratory Data Analysis
        This tab offers an exploratory analysis of the uploaded dataset, presenting key statistics for numerical variables such as minimum, maximum, mean, median, and quartiles. It also displays the distribution of categorical variables (Sex and Diabetes Diagnosis) using bar charts, and histograms for continuous variables.

        #### Performance Results
        Displays the predictive performance of the model using the uploaded data. This includes a confusion matrix and performance metrics like accuracy, precision, sensitivity, and F1 score.

        #### Model Inspection
        Features graphs that inspect the model's key attributes, including permutation importance and SHAP values, both overall and separated by class. This tab also explores the combined importance from these techniques and the significance of different attribute types.


        ### Data Requirements
        Please ensure that the dataset uploaded is in CSV format and contains the following variables with their respective units:

        - **Age** (years)
        - **Urea** (mmol/L)
        - **Cr** (Creatinine, µmol/L)
        - **Chol** (Total cholesterol, mmol/L)
        - **TG** (Triglycerides, mmol/L)
        - **HDL** (High-density lipoprotein, mmol/L)
        - **LDL** (Low-density lipoprotein, mmol/L)
        - **VLDL** (Very low-density lipoprotein, mmol/L)
        - **BMI** (Body Mass Index, kg/m²)
        - **Sex** ('Female' or 'Male')
        - **Diagnosis** ('Diabetes', 'Non-diabetes', 'Prediabetes')

        ### Model Performance
        The used model is a **XGBoost classifier**, which achieved the highest performance on the testing data in our experiments, showing a macro average **F1-score** of **0.792**.
                    
        ### Test the Application
        You can test the application using a sample dataset available at [this link](https://github.com/colme99/multiclass_diabetes_prediction/blob/main/data/test_data.csv).
        """)



def main():


    # Set the configuration of the page (title, colors)
    config_page_settings()

    # Create a sidebar and set the title
    st.sidebar.title("Navigation")

    # Create buttons in the sidebar to select the tab
    tabs = st.sidebar.radio("Select a tab", ["Introduction", "Data upload", "Exploratory Analysis", "Performance Results", "Model Inspection"])

    # Set the default values for the variables in the session
    initialize_state_variables()

    # Data upload tab
    if tabs == "Introduction":

        # Set the title
        st.title("Introduction")

        # Show information about the app and the expected data format
        show_introduction_explanation()

    # Data upload tab
    elif tabs == "Data upload":

        # Message for the user to upload the data
        st.title("Upload the dataset")

        # Button to upload the data
        uploaded_file = st.file_uploader("Upload the dataset in CSV format", type = "csv")

        # Perform the data preprocessing
        if uploaded_file is not None:
            perform_data_preprocessing(uploaded_file)

    # Exploratory data analysis tab
    elif tabs == "Exploratory Analysis":

        # Set the title
        st.title("Exploratory Analysis")

        # If the data is uploaded
        if st.session_state.uploaded_file is not None:

            # Perform the exploratory data analysis
            perform_exploratory_data_analysis()

        else:
            # If the data is not uploaded, inform the user
            st.warning("No data loaded. Please upload a the dataset in the 'Data upload' tab.")

    # Performance results tab
    elif tabs == "Performance Results":

        # Set the title
        st.title("Performance Results")

        # If the data is uploaded
        if st.session_state.uploaded_file is not None:

            # Show the predictive performance results
            evaluate_model(st.session_state.pipeline, st.session_state.X_test, st.session_state.y_test, st.session_state.diagnosis_encoder)

        else:
            # If the data is not uploaded, inform the user
            st.warning("No data loaded. Please upload a the dataset in the 'Data upload' tab.")

    # Model inspection tab
    elif tabs == "Model Inspection":

        # Set the title
        st.title("Model Inspection")

        # If the data is uploaded
        if st.session_state.uploaded_file is not None:

            # Perform the model inspection
            perform_model_inspection()

        else:
            # If the data is not uploaded, inform the user
            st.warning("No data loaded. Please upload a the dataset in the 'Data upload' tab.")
        
    else:
        # Ask the user to upload the data
        st.write('Please upload the dataset in CSV format.')



if __name__ == "__main__":
    main()
