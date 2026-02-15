Although the model achieved a high R² score (0.99), comparison with a naive baseline(0.992) predictor revealed that the baseline slightly outperformed the trained model. This indicates that most predictive power arises from strong autocorrelation between consecutive closing prices, rather than complex feature interactions. Therefore, the model does not meaningfully improve upon a simple persistence strategy.

The correlation matrix reveals extremely strong linear relationships among the price-based features (Open, High, Low, Close, MA5, and MA10), with correlation coefficients exceeding 0.99 in most cases. This indicates severe multicollinearity, meaning that these variables contain nearly identical information and move almost perfectly together. Such redundancy suggests that the moving averages (MA5 and MA10) add little new information beyond the current closing price, as they are directly derived from it. Furthermore, the correlation between the current Close price and the Target variable (next-day Close) is approximately 0.999, demonstrating very strong short-term autocorrelation in stock prices. This implies that tomorrow’s closing price is highly predictable from today’s price, which explains why a naive baseline predictor performs nearly as well as the trained model. In contrast, Volume exhibits a moderate negative correlation (around −0.60) with price features, indicating that trading activity behaves somewhat independently from price levels. Overall, the matrix suggests that most predictive power in the model arises from price persistence rather than complex interactions among distinct features, highlighting both the stability of short-term price movements and the redundancy within the engineered feature set.

Because Close is almost perfectly correlated with Target, adding more raw price features won’t help.

You need something that captures:
	•	Momentum
	•	Volatility
	•	Relative change
	•	Market behavior

NOT absolute price.
# Stock_price-_pred1
