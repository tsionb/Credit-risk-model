# Credit-risk-model

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Accord fundamentally changed credit risk management by directly linking a bank's internal risk models to its minimum capital requirements. This creates a strong regulatory requirement for models that are not only accurate but also:

   -Transparent and Explainable: Supervisors must validate these models, requiring clear justification for variable selection, methodology, and performance metrics

   -Well-documented: Comprehensive documentation covering data sources, assumptions, and limitations is mandatory

   -Auditable: Models must be structured for easy review by regulators and internal auditors

An opaque "black-box" model, regardless of its predictive power, risks regulatory rejection and could result in higher capital charges. Therefore, interpretability is not merely a technical preference but a prerequisite for regulatory compliance and capital efficiency.

### 2. Proxy Variables and Business Risks
2.	A direct "default" label is often unavailable due to low default portfolios, legal delays in finalizing defaults, or the need for a consistent, leading indicator. Thus, creating a proxy variable (e.g., 90+ days past due) is necessary to build statistical models. However, this introduces significant business risks:
   
  •	Model Bias: The proxy may not perfectly correlate with true economic default. A model optimized for predicting 90-day delinquency might perform poorly in identifying borrowers who default without a long delinquency period (e.g., via sudden bankruptcy).
  
  •	Misaligned Incentives: If the proxy is gameable (e.g., encouraging short-term forbearance to avoid a 90-day flag), it can lead to hidden risk accumulation.
  
  •	Inaccurate Risk Pricing & Capital Calculation: An imperfect proxy distorts the estimated Probability of Default (PD), leading to mispriced loans and incorrect regulatory capital, ultimately affecting profitability and stability.


### 3. Model Selection Trade-offs
3.	The choice between a simple model (e.g., Logistic Regression with Weight of Evidence) and a complex one (e.g., Gradient Boosting) involves a core trade-off between explainability and predictive power.
   
 •	Simple, Interpretable Models (e.g., Logistic Regression):
 
  o	Advantages: High explainability. Coefficients directly show a variable's impact, enabling clear "reason codes" for adverse actions (e.g., "denied due to high debt-to-income ratio"). This aligns with regulatory expectations (e.g., SR 11-7) and fair lending compliance. Easier to validate, document, and implement in legacy systems.
  
  o	Disadvantage: May sacrifice predictive accuracy, especially with complex, non-linear relationships in novel data (e.g., transaction records as in the RFMS paper).
  
 •	Complex, High-Performance Models (e.g., Gradient Boosting):
 
  o	Advantages: Often superior predictive accuracy, potentially capturing subtle patterns in alternative data (e.g., bank card transactions), leading to better risk discrimination and profit.
  
  o	Disadvantages: "Black-box" nature makes explaining individual decisions difficult, raising regulatory, reputational, and fairness concerns. Requires extensive documentation and sophisticated validation frameworks to gain supervisory approval.
  
 Conclusion: In a regulated context, the regulatory demand for explainability and auditability under Basel II often outweighs marginal gains in accuracy. A well-specified, interpretable model that meets supervisory standards is typically preferred. The trend involves using complex models for initial insight and feature discovery, but often finalizing with simpler, more interpretable models or investing heavily in post-hoc explainability techniques to bridge the gap.
