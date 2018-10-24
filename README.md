# Recurrent-Reinforcement-Learning-in-Bitcoin-Trading
Final Thesis at Fudan University, built a trading strategy on Bitcoin market using recurrent reinforcement learning

- Environment: python3.6
- Ouput in dir results and signal_results, which should be built manually.
- Abstract: This paper presents methods to trade on Bitcoin market (GDAX) through Recurrent Reinforcement Learning. Unlike supervised learning with labels, Reinforcement Learning learns the parameters of the model through trail and error, that is, through trading and optimizing the objective function repeatedly. We find it’s better to adopt a moving window when training the parameters, with the Sharpe Ratio 2.75. To further optimize the model, we adopt technical indicators like MA, MTM. Results show the strategy can increase the Sharpe Ratio as well as decrease the Maximum Drawdown to 1.35%. Finally as Bitcoin is allowed to be shorted since last year, we test our origin model with short sell- ing permitted. The model achieves better performance, with the annual return 112.00%, which further proves the effectiveness of the model.
- Keywords: Reinforcement Learning, Quantitative Trading, Bitcoin, Technical Indicators
