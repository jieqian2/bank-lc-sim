# Abstract

The main idea of this project is that we use data mining techniques to extract effective information from real-life databases and utilize this information to form investment strategies of a synthetic retail bank. By analyzing the loan information of Lending Club database, we developed several machine learning classifiers and implement them with various portfolio management techniques in our simulation of 40 investment cycles. Built upon the previous work of the IBM team, we have improved and redesigned the data preprocessing process, default detection models, investment mechanism, portfolio management strategies and simulation parameters. In addition, we have experimented adding more features and factors to our algorithm, such as prepayment detection models. After comparing multiple sets of parameters, we conclude that XGBoost model using under sampling technique generates the best performance. Even though actively portfolio management brings little benefits in our simulation, it will require further study on tuning the parameters to confirm its ineffectiveness.
 
# Background

Lending Club is an American peer-to-peer lending company. It enables borrowers to create unsecured personal loans between $1,000 and $40,000. The standard loan period is three years and can be extend to five years. Investors can search and browse the loan listings on Lending Club website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from interest. Lending Club makes money by charging borrowers an origination fee and investors a service fee.

Lending Club database contains millions of loans, in this project, we use these data to dig up and analyze the behavior characters of borrowers, train different machine learning models, such as Random Forest, Logistic Regression, neural network, to estimate the prepay and default probability of each loans. 

With these models, we simulate a retail bank, experiment with optimization approaches, implement different dynamic investment strategies and analyze outcomes, to guide the bank's lending and borrowing strategies over time, and to report the risk and rewards of our experiments.
 
# Data preprocessing

## Database and its Distribution

The raw Lending Club (LC) database details 2,260,668 loans issued by Lending Club from June 2007 to December 2018. The loan status is shown in Table 1-1.
Table 1-1 Lending Club Loans Description
Loan Status	Fully paid	Charged off	Late (16-30 days)	Late (31-120 days)	Current
Num of Loans	1,041,952	261,655	3,737	21,897	919,695
Loan Status	In Grace Period	Default
	Violation of Credit Policies.
Status: Charged Off	Violation of Credit Policies.
Status: Fully Paid	
Number	8,952	31	761	1,988	

Obviously, ‘Current’, 'Fully paid' and 'Charged off' loans account for the most important part of the data set. 
