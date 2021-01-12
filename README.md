# Project Overview

This project provides a consolidated view of data for all outbreaks between 1996 to 2020 through an interactive dashboard, calculates CFR (Case Fatality Ratio) for outbreaks and aims to model COVID-19 using a mathematical model called SIR (Susceptibility, Infectivity and Recoverablity).



# Project Instructions
## Install
This project requires **Python 3.x** and the following Python libraries installed:

 - NumPy
 - Pandas
 - scikit-learn
 - spaCy
 - BeautifulSoup
 - pycountry
 - matplotlib
 - word2num
 - seaborn
 - scipy

 
 **Microsoft PowerBI** should also be installed to view and interact with the dashboard.


## Instructions
### 1.  To view the Dashboard
#### Using PowerBI Desktop:
1. Clone the repository and navigate to the downloaded folder. <br/>
	`git clone https://csil-git1.cs.surrey.sfu.ca/kdesai/outbreak.git` <br/>
	`cd outbreak`
2. Open the dashboard `Disease_Outbreak_Final_Dashboard.pbix` using Microsoft PowerBI.

#### Using PowerBI Online:
Click on the following link to view the dashboard:
[Power BI Dashboard](https://app.powerbi.com/groups/me/reports/2efa597c-fcd4-4b30-bf50-15a11d6a25f6/ReportSection7f32491e622c28242dea?ctid=4a1e5cee-f43e-451d-b150-1486f954ef55)

**Note: SFU Microsoft Office Administrator has disabled the public access to PowerBI reports/dashboards and data. To view the reports/dashboards, please log into PowerBI.**

###  2. To generate SIR prediction plots for US and Canada
1.  Navigate to SIR_Model folder. <br/>
	`cd SIR_Model`
2. To generate features for predicting transmission and recovery rate, run `SIR_feature_generation.py` using the following commands:

	For US: <br/>
`python SIR_feature_generation.py covid_19_data.csv -c US` <br/>
	For Canada: <br/>
`python SIR_feature_generation.py covid_19_data.csv -c Canada`<br/>

	**Note: The SIR model is trained and tested on US and Canada.**
3. To generate the prediction plots, run the following commands: <br/>
	`python Modelling_US.py` <br/>
	`python Modelling_Canada.py` <br/>





	
