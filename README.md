# 📊 Ecommerce Customer Spending Prediction

This project builds a **Linear Regression model** to predict the **Yearly Amount Spent** by ecommerce customers based on their behavior (session length, time on app, website usage, membership length).  

---

# Project Overview
- Dataset: **Ecommerce Customers.csv**  
- Goal: Predict how much a customer will spend yearly using regression techniques.  
- Tools: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn  

---

# 🔑 Features Used
- **Avg. Session Length** – Average duration of a customer session.  
- **Time on App** – Time spent on mobile app.  
- **Time on Website** – Time spent on website.  
- **Length of Membership** – How long the customer has been a member.  

Target variable:
- **Yearly Amount Spent**

---

#📂 Project Workflow
1. **Data Loading & Exploration**
   - Loaded CSV dataset using Pandas.  
   - Performed EDA with Seaborn (`pairplot`, `jointplot`, `lmplot`).  

2. **Feature Selection**
   - Selected independent variables (`Avg. Session Length`, `Time on App`, `Time on Website`, `Length of Membership`).  
   - Dependent variable: `Yearly Amount Spent`.  

3. **Data Preprocessing**
   - Split dataset into training and testing sets (70% / 30%).  
   - Standardized features using `StandardScaler`.  

4. **Model Training**
   - Applied **Linear Regression**.  
   - Evaluated using 5-fold Cross Validation (MSE as metric).  

5. **Prediction & Evaluation**
   - Predicted on test set.  
   - Compared predicted vs actual values.  
   - Evaluated with **R² Score**.  
   - Visualized prediction errors using KDE plot.  

---

# Results
- Model predicts yearly spending with good accuracy.  
- **Evaluation Metrics:**  
  - Cross-validation MSE (negative values reported by sklearn).  
  - R² score printed after evaluation.  

---

 #Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

# How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ecommerce-linear-regression.git
