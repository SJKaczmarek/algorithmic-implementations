# Algorithmic Implementations

This folder contains code implementations of various algorithms, exploring different areas of machine learning and artificial intelligence. 

## Published Implementations:

1. **xlstmtime_model.py**

   This Python file provides a PyTorch implementation of the xLSTMTime model, a novel architecture for long-term time series forecasting (LTSF). xLSTMTime leverages the strengths of the extended Long Short-Term Memory (xLSTM) model, incorporating exponential gating and a revised memory structure to effectively capture temporal dependencies in time series data. 

   A detailed review of xLSTMTime, including its architecture, performance benchmarks, and potential impact on the LTSF landscape, can be found in the article: [xLSTMTime: A Competitive Recurrent Architecture for Long-Term Time Series Forecasting](https://www.linkedin.com/pulse/xlstmtime-competitive-recurrent-architecture-time-series-kaczmarek-jdmpe/).

   **Explanation:**

   *   **Import Libraries:** Imports necessary libraries including PyTorch.
   *   **sLSTMCell:**
       *   This class defines the sLSTM cell, a variant of LSTM with exponential gating and stabilization.
       *   `__init__`: Initializes weights and biases for linear transformations.
       *   `forward`: Implements the forward pass of the sLSTM cell, calculating hidden state, cell state, normalization state, and memory based on input, previous hidden state, and previous cell state.
   *   **mLSTMCell:**
       *   This class defines the mLSTM cell, using matrix memory for increased capacity.
       *   `__init__`: Initializes weights and biases for linear transformations used in query, key, value, and gate calculations.
       *   `forward`: Implements the forward pass of the mLSTM cell, updating matrix memory, normalization, and hidden state.
   *   **xLSTMTime:**
       *   This class defines the overall xLSTMTime model for time series forecasting.
       *   `__init__`:
           *   Initializes parameters like input size, hidden size, output size, sequence length, and whether to use mLSTM.
           *   Defines layers for series decomposition, linear transformations, xLSTM block (sLSTM or mLSTM), and instance normalization.
       *   `forward`:
           *   Implements the forward pass of the model.
           *   Performs series decomposition to extract trend and seasonal information.
           *   Applies linear transformation and batch normalization to the input.
           *   Iterates through the sequence length, applying either the sLSTM or mLSTM cell based on the `use_mlstm` flag.
           *   Applies a final linear transformation and instance normalization to produce the output.
   *   **Example Usage:**
       *   Demonstrates how to create an instance of the xLSTMTime model and perform a forward pass with sample input data.

   **Note:** This implementation is a framework and may require further adjustments and hyperparameter tuning based on specific datasets and tasks. The code assumes basic familiarity with PyTorch and deep learning concepts.

2. **Stay Tuned for More** 

   This repository will continue to expand with new algorithmic implementations. Follow me on [GitHub](https://github.com/SJKaczmarek/) and [LinkedIn](https://www.linkedin.com/in/sylvesterkaczmarek/) to stay updated on the latest additions.

   Have a specific algorithm you'd like to see implemented? Feel free to submit a request! 

## License

This repository is licensed under the [MIT License](LICENSE).

## Contact

For sensitive matters, inquiries, or professional collaborations, please reach out via email at [space.stranger698@8shield.net](mailto:space.stranger698@8shield.net). For quicker responses, you can also connect with me on my [LinkedIn Profile](https://www.linkedin.com/in/sylvesterkaczmarek/).
