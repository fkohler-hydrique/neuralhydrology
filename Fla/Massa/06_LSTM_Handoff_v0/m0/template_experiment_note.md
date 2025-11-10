# Experiment: [Experiment Name]

## Hypothesis

[Clearly state the hypothesis you are testing in this experiment. What do you expect to happen, and why? Be specific.]

*Example: "Applying a log-transform to the target variable (streamflow) will improve the model's ability to predict high-flow events, resulting in higher NSE scores during peak flow periods."*

## Configuration

*   **Model Type:** [e.g., SequentialForecastLSTM, HandoffForecastLSTM]
*   **Config File:** [`config.yml`](link/to/your/config.yml)
*   **Key Parameters:**
    *   `seq_length`: [Value]
    *   `predict_last_n`: [Value]
    *   `hindcast_inputs`: [List of inputs]
    *   `forecast_inputs`: [List of inputs]
    *   `target_scaler`: [e.g., StandardScaler, LogScaler]
    *   `lstm_dropout`: [Value, if applicable]
    *   `state_handoff_network`: [Brief description or link to config]

## Data

*   **Dataset:** [Name of the dataset used]
*   **Time Period:** [Start date] - [End date]
*   **Preprocessing:** [Briefly describe any data preprocessing steps applied, e.g., scaling, transformations, handling of missing values.]

## Experiment Tracking

*   **Tracking Tool:** [e.g., MLflow, Weights & Biases]
*   **Run ID:** [Link to the specific run in your tracking tool]

## Results

[Summarize the key results of the experiment. Include relevant metrics (NSE, KGE, RMSE, etc.) and any observations about the model's behavior.]

*   **Key Metrics:**
    *   NSE (Test): [Value]
    *   KGE (Test): [Value]
    *   RMSE (Test): [Value]
*   **Visualizations:** [Link to any relevant plots or visualizations]

## Conclusion

[State whether the results support or refute your hypothesis. Discuss any insights gained from the experiment and potential next steps.]

*Example: "The results partially support the hypothesis. The log-transformed model did improve NSE during high-flow events, but it slightly reduced overall KGE. Next steps include exploring different regularization techniques to improve the model's generalization performance."*

## Notes

[Add any additional notes or observations about the experiment, such as challenges encountered, unexpected behavior, or ideas for future experiments.]
