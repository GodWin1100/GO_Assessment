# %% [markdown]
# # Python Assessment
# 

# %% [markdown]
# ## Importing Data
# 

# %%
import pandas as pd


# %%
df_employee = pd.read_csv("employee_data.csv")
print(f"Shape: {df_employee.shape}")
df_employee.head()


# %%
df_insurance = pd.read_csv("insurance_data.csv")
print(f"Shape: {df_insurance.shape}")
with pd.option_context("display.max_columns", 0):
    display(df_insurance.head())


# %%
df_insurance.columns.sort_values()


# %%
df_vendor = pd.read_csv("vendor_data.csv")
print(f"Shape: {df_vendor.shape}")
df_vendor.head()


# %% [markdown]
# ## Task 1
# 
# - Merge the 3 dataset and create 1 view of data.
# - You can merge `insurance_data.csv` and `employee_data.csv` on `AGENT_ID`
# - You can merge `insurance_data.csv` and `vendor_data.csv` on `VENDOR_ID`
# - **Note:** Use left Outer join as not all claims require Vendor
# 

# %%
df_merged = df_insurance.merge(df_employee, on="AGENT_ID", suffixes=["_INSURANCE", "_AGENT"]).merge(
    df_vendor, on="VENDOR_ID", how="left"
)
print(f"Shape: {df_merged.shape}")
with pd.option_context("display.max_columns", 0):
    display(df_merged.head())


# %%
df_merged.columns.sort_values()


# %% [markdown]
# ## Task 2
# 
# - Business Leader wants to find **Top 3 insurance Type** where we are getting `most insurance claims`?
# 

# %%
# df_insurance["INSURANCE_TYPE"].value_counts()[:3]
df_insurance["INSURANCE_TYPE"].value_counts().nlargest(3)


# %%
df_insurance["CLAIM_STATUS"].unique()


# %%
# Claim Status Accepted i.e. getting insurance claim
claim_condition = df_insurance["CLAIM_STATUS"] == "A"
# df_insurance[claim_condition]["INSURANCE_TYPE"].value_counts()[:3]
df_insurance[claim_condition]["INSURANCE_TYPE"].value_counts().nlargest(3)


# %% [markdown]
# Top 3 Insurance Type with `Accepted/Approved` status and `Over All` Insurance Claims
# 
# 1. Property
# 2. Mobile
# 3. Health
# 

# %%
df_insurance[["INSURANCE_TYPE", "CLAIM_STATUS"]].reset_index().groupby(
    ["INSURANCE_TYPE", "CLAIM_STATUS"]
).count().sort_values(by="index", ascending=False)


# %% [markdown]
# ## Task 3
# 
# - Business Leader wants to find **Top 5 States** where we are getting most insurance claims for customer belonging to `HIGH(H)` risk segment?
# 

# %%
df_insurance["RISK_SEGMENTATION"].unique()


# %%
# High Risk Segmentation
high_risk_condition = df_insurance["RISK_SEGMENTATION"] == "H"  # & (df_insurance["CLAIM_STATUS"] == "A")
df_insurance[high_risk_condition]["STATE"].value_counts().nlargest(5)


# %% [markdown]
# **Top 5 States** with `High` risk segmentation insurance
# 
# 1. CA
# 2. AZ
# 3. FL
# 4. TN
# 5. AR
# 

# %% [markdown]
# ## Task 4
# 
# - Business wants to create a new variable `“COLOCATION”` which will have following values
#   > IF Customer State == Incident State == Agent Address State  
#   > THEN 1  
#   > ELSE 0
# - Find the mean of this new column
# 

# %%
df_ins_emp = df_merged.copy()
df_ins_emp.columns.sort_values()


# %%
# other option: np.where
df_ins_emp["COLOCATION"] = 0  # setting all as 0
colocation_condition = (df_ins_emp["STATE_INSURANCE"] == df_ins_emp["INCIDENT_STATE"]) & (
    df_ins_emp["INCIDENT_STATE"] == df_ins_emp["STATE_AGENT"]
)
df_ins_emp.loc[colocation_condition, "COLOCATION"] = 1  # setting 1 if condition is met
# df_ins_emp["COLOCATION"] = df_ins_emp["COLOCATION"].mask(colocation_condition, 1).sample(10, random_state=6)
print(f'Mean:{df_ins_emp["COLOCATION"].mean()}')
with pd.option_context("display.max_columns", 0):
    display(df_ins_emp[["STATE_INSURANCE", "INCIDENT_STATE", "STATE_AGENT", "COLOCATION"]].sample(10, random_state=6))


# %%
df_ins_emp["COLOCATION_2"] = colocation_condition.astype(int)  # Boolean are Sub-group of numeric data-type
print(f'Mean:{df_ins_emp["COLOCATION_2"].mean()}')
with pd.option_context("display.max_columns", 0):
    display(
        df_ins_emp[["STATE_INSURANCE", "INCIDENT_STATE", "STATE_AGENT", "COLOCATION", "COLOCATION_2"]].sample(
            10, random_state=6
        )
    )


# %% [markdown]
# Mean of Colocation: 0.0044 = 0.44%
# 

# %% [markdown]
# ## Task 5
# 
# - Data entry error was detected in the data and you are required to correct it. If for any claim transaction `“AUTHORITY_CONTACTED”` is NOT `“Police”` and `POLICE_AVAILABLE == 1` Then Update `“AUTHORITY_CONTACTED”` to `"Police"`.
# 

# %%
df_ins_corrected = df_insurance.copy()  # copy of data to avoid mutation
df_ins_corrected[["AUTHORITY_CONTACTED", "POLICE_REPORT_AVAILABLE"]].value_counts()


# %%
contacted_condition = (df_ins_corrected["AUTHORITY_CONTACTED"] != "Police") & (
    df_ins_corrected["POLICE_REPORT_AVAILABLE"]
)
df_ins_corrected.loc[contacted_condition, "AUTHORITY_CONTACTED"] = "Police"
df_ins_corrected[["AUTHORITY_CONTACTED", "POLICE_REPORT_AVAILABLE"]].value_counts()


# %% [markdown]
# ## Task 6
# 
# - Business wants to check the Claim Amount for deviation for each transaction, they would like you to calculate as follow
#   > CLAIM_DEVIATION = AVG_CLAIM_AMOUNT_FOR_LAST_30DAYS (same insurance type) / CURRENT_CLAIM_AMOUNT
# - If the `value < 0.5` THEN `CLAIM_DEVIATION = 1` ELSE `0`
# - If there is less than 30 days of transaction history THEN -1
# - **Note:** LAST_30DAYS does not include current day
# 

# %% [markdown]
# Static date approach in Fiddle > Task 6
# 

# %%
df_claim_deviation = df_insurance[["TXN_DATE_TIME", "INSURANCE_TYPE", "CLAIM_AMOUNT"]].copy()
df_claim_deviation["TXN_DATE_TIME"] = pd.to_datetime(df_claim_deviation["TXN_DATE_TIME"])
df_claim_deviation.describe(datetime_is_numeric=True)


# %%
def claim_deviations(row):
    current_date = row["TXN_DATE_TIME"]
    df_window = current_date - df_claim_deviation["TXN_DATE_TIME"]  # get series with time_delta
    window_condition = (df_window <= pd.to_timedelta("30d")) & (
        df_window > pd.to_timedelta("0d")
    )  # last 30 days condition where current date is excluded
    df_window_30 = df_claim_deviation[window_condition]
    avg_claim = df_window_30.groupby("INSURANCE_TYPE")["CLAIM_AMOUNT"].mean()  # calculating mean wrt insurance type
    # if less than 30 days transaction for particular type then return -1
    # if less than 30 days transaction criteria is not for particular type then
    # .get(row["INSURANCE_TYPE"], 0) is not required 
    if df_window_30.groupby("INSURANCE_TYPE")["TXN_DATE_TIME"].nunique().get(row["INSURANCE_TYPE"], 0) < 30:
        return -1
    return 1 if avg_claim[row["INSURANCE_TYPE"]] / row["CLAIM_AMOUNT"] < 0.5 else 0


df_claim_deviation["CLAIM_DEVIATION"] = df_claim_deviation.apply(claim_deviations, axis=1)


# %%
df_claim_deviation.sample(10, random_state=26)


# %%
df_claim_deviation["CLAIM_DEVIATION"].value_counts()


# %% [markdown]
# ## Task 7
# 
# - Find All Agents who have worked on `more than 2 types of Insurance Claims`. Sort them by `Total Claim Amount Approved` under them in `descending order`.
# 

# %%
df_agents = df_merged[["AGENT_ID", "AGENT_NAME", "INSURANCE_TYPE", "CLAIM_AMOUNT", "CLAIM_STATUS"]].copy()
df_agents.head()


# %%
multi_type_condition = (
    df_agents.groupby("AGENT_ID")["INSURANCE_TYPE"].nunique() > 2
)  # More than 2 Insurance Type of Agent
multi_type_agent = multi_type_condition[multi_type_condition].index
multi_type_agent


# %%
# condition with Approved Claim Status and Multi Insurance Type Agent
multi_approved_condition = (df_agents["AGENT_ID"].isin(multi_type_agent)) & (df_agents["CLAIM_STATUS"] == "A")
df_agents[multi_approved_condition].groupby(["AGENT_ID", "AGENT_NAME"]).sum().sort_values(
    "CLAIM_AMOUNT", ascending=False
)


# %% [markdown]
# ## Task 8
# 
# - `Mobile & Travel Insurance` premium are `discounted by 10%`
# - `Health and Property Insurance` premium are `increased by 7%`
# - `Life and Motor Insurance` premium are marginally `increased by 2%`
# - What will be `overall change in % of the Premium Amount Collected` for all these Customer?
# 

# %%
df_premium = df_insurance[["INSURANCE_TYPE", "PREMIUM_AMOUNT"]].copy()
df_premium["INSURANCE_TYPE"].unique()


# %%
df_premium_change = df_premium.groupby("INSURANCE_TYPE").sum()
df_premium_change["NEW_PREMIUM_AMT"] = df_premium_change["PREMIUM_AMOUNT"]
df_premium_change


# %%
df_premium_change.loc[["Mobile", "Travel"], "NEW_PREMIUM_AMT"] = (
    df_premium_change.loc[["Mobile", "Travel"], "PREMIUM_AMOUNT"] * 0.9
)  # discounted by 10%
df_premium_change.loc[["Health", "Property"], "NEW_PREMIUM_AMT"] = (
    df_premium_change.loc[["Health", "Property"], "PREMIUM_AMOUNT"] * 1.07
)  # increased by 7%
df_premium_change.loc[["Life", "Motor"], "NEW_PREMIUM_AMT"] = (
    df_premium_change.loc[["Life", "Motor"], "PREMIUM_AMOUNT"] * 1.02
)  # increased by 2%
df_premium_change


# %%
df_premium_change.sum().pct_change()

# %%
prem_amt, new_prem_amt = df_premium_change.sum()
pct_change = ((new_prem_amt - prem_amt) / prem_amt) * 100
print(f"Total Premium Amt: {prem_amt:.2f}\nTotal New Premium Amt: {new_prem_amt:.2f}\n% Change: {pct_change:.5f}%")


# %% [markdown]
# Overall change in % of the Premium Amount Collected: 2.679%
# 

# %% [markdown]
# ## Task 9
# 
# - Business wants to give discount to customer who are loyal and under stress due to Covid 19. They have laid down an eligibility Criteria as follow
# - IF `CUSTOMER_TENURE > 60 AND EMPLOYMENT_STATUS = “N” AND NO_OF_FAMILY_MEMBERS >=4` THEN `1` ELSE `0`
# - Create a new column `“ELIGIBLE_FOR_DISCOUNT”` and find it
#   `mean`.
# 

# %%
df_eligible = df_insurance[["TENURE", "EMPLOYMENT_STATUS", "NO_OF_FAMILY_MEMBERS"]].copy()
eligible_condition = (
    (df_eligible["TENURE"] > 60)
    & (df_eligible["EMPLOYMENT_STATUS"] == "N")
    & (df_eligible["NO_OF_FAMILY_MEMBERS"] >= 4)
)
df_eligible["ELIGIBLE_FOR_DISCOUNT"] = eligible_condition.astype(int)
df_eligible["ELIGIBLE_FOR_DISCOUNT"].mean()


# %% [markdown]
# Mean for discount eligibility customer: 0.0299
# 

# %% [markdown]
# ## Task 10
# 
# - Business wants to check `Claim Velocity` which is defined as follow
#   > CLAIM_VELOCITY = NO_OF_CLAIMS_IN_LAST30DAYS (for the current insurance type) / NO_OF_CLAIMS_IN_LAST3DAYS (for the current insurance type)
# - **Note**: LAST30DAYS & LAST3DAYS does not include current
#   day
# 

# %%
df_insurance.columns.sort_values()


# %%
df_claim_velocity = df_insurance[["TXN_DATE_TIME", "INSURANCE_TYPE", "CLAIM_STATUS"]].copy()
df_claim_velocity["TXN_DATE_TIME"] = pd.to_datetime(df_claim_velocity["TXN_DATE_TIME"])


# %%
import numpy as np


def claim_velocities(row):
    current_date = row["TXN_DATE_TIME"]
    df_window = current_date - df_claim_velocity["TXN_DATE_TIME"]  # get series with time_delta
    window_30_condition = (df_window <= pd.to_timedelta("30d")) & (
        df_window > pd.to_timedelta("0d")
    )  # last 30 days condition where current date is excluded
    df_window_30 = df_claim_velocity[window_30_condition]
    window_3_condition = (df_window <= pd.to_timedelta("3d")) & (
        df_window > pd.to_timedelta("0d")
    )  # last 3 days condition where current date is excluded
    df_window_3 = df_claim_velocity[window_3_condition]
    # If claim_status = A is required we can pass filter as
    # df_window_3X[df_window_3X['CLAIM_STATUS']=='A'].groupby("INSURANCE_TYPE")["CLAIM_STATUS"].count().get(row["INSURANCE_TYPE"],np.nan)
    claim_count_30 = (
        df_window_30.groupby("INSURANCE_TYPE")["CLAIM_STATUS"].count().get(row["INSURANCE_TYPE"], np.nan)
    )  # calculating no of insurance claim wrt to insurance type
    claim_count_3 = (
        df_window_3.groupby("INSURANCE_TYPE")["CLAIM_STATUS"].count().get(row["INSURANCE_TYPE"], np.nan)
    )  # calculating no of insurance claim wrt to insurance type
    return claim_count_30 / claim_count_3


df_claim_velocity["CLAIM_VELOCITY"] = df_claim_velocity.apply(claim_velocities, axis=1)


# %%
df_claim_velocity.sample(10, random_state=14)


# %% [markdown]
# ## Task 11
# 
# - Find all low performing agents i.e. employees who are in the bottom 5 percentile based on Claims worked by them.
# 

# %%
df_performance = df_insurance[["AGENT_ID", "CLAIM_AMOUNT"]].copy()
df_performance.head()


# %% [markdown]
# Based on No. of Claims
# 

# %%
agent_performance = df_performance.groupby("AGENT_ID").count()
quantile_5 = agent_performance.quantile(0.05).values[0]
res_agent_performance = agent_performance[agent_performance["CLAIM_AMOUNT"] < quantile_5]
print(f"Total Entries: {res_agent_performance.shape[0]}")
res_agent_performance.rename({"CLAIM_AMOUNT": "NO_OF_CLAIMS"}, axis=1)


# %% [markdown]
# Based on Claim Amount
# 

# %%
agent_performance = df_performance.groupby("AGENT_ID").sum()
quantile_5 = agent_performance.quantile(0.05).values[0]
res_agent_performance = agent_performance[agent_performance["CLAIM_AMOUNT"] < quantile_5]
print(f"Total Entries: {res_agent_performance.shape[0]}")
res_agent_performance


# %% [markdown]
# ## Task 12
# 
# - Business wants to find all Suspicious Employees (Agents).
# - IF `TOTAL CLAIM AMOUNT` which meet below criteria is `>= 15000` THEN AGENT is classified as `Suspicious` ELSE `Not`
# - `CLAIM_STATUS = Approved` AND `CUSTOMER_RISK_SEGMENTATION = High` AND `INCIDENT_SEVERITY = “Major Loss”`
# - If `Suspicious`, THEN `1` ELSE `0`.
# - Find `mean` of this column.
# 

# %%
df_suspicious = df_insurance[
    ["AGENT_ID", "CLAIM_AMOUNT", "CLAIM_STATUS", "RISK_SEGMENTATION", "INCIDENT_SEVERITY"]
].copy()
df_suspicious.head()


# %%
df_suspicious["CLAIM_STATUS"].unique()


# %%
df_suspicious["RISK_SEGMENTATION"].unique()


# %%
df_suspicious["INCIDENT_SEVERITY"].unique()


# %%
suspicious_condition = (
    (df_suspicious["CLAIM_STATUS"] == "A")
    & (df_suspicious["RISK_SEGMENTATION"] == "H")
    & (df_suspicious["INCIDENT_SEVERITY"] == "Major Loss")
)
df_suspicious_group = df_suspicious[suspicious_condition].groupby("AGENT_ID").sum()
suspicious_agent = df_suspicious_group[df_suspicious_group["CLAIM_AMOUNT"] >= 15000].index
suspicious_agent


# %%
df_suspicious_employee = df_employee.copy()
df_suspicious_employee["SUSPICIOUS"] = 0
df_suspicious_employee.loc[df_suspicious_employee["AGENT_ID"].isin(suspicious_agent), "SUSPICIOUS"] = 1
df_suspicious_employee.head()


# %%
df_suspicious_employee["SUSPICIOUS"].value_counts()


# %%
mean_suspicious = df_suspicious_employee["SUSPICIOUS"].mean()
print(f"Mean Suspicious: {mean_suspicious:.05f}")


# %% [markdown]
# Mean of Suspicious Employee: 0.12417
# 

# %% [markdown]
# # Fiddle
# 

# %% [markdown]
# ## Task 6
# 

# %% [markdown]
# Static Date  
# Assuming Current Date as `2021-03-24`
# 

# %%
dfid_claim_deviation = df_insurance[["TXN_DATE_TIME", "INSURANCE_TYPE", "CLAIM_AMOUNT"]].copy()
dfid_claim_deviation["TXN_DATE_TIME"] = pd.to_datetime(dfid_claim_deviation["TXN_DATE_TIME"])
current_date = pd.to_datetime("2021-03-24")
df_window = current_date - dfid_claim_deviation["TXN_DATE_TIME"]
window_condition = (df_window <= pd.to_timedelta("30d")) & (df_window > pd.to_timedelta("0d"))
df_window_30 = dfid_claim_deviation[window_condition]
avg_claim = df_window_30.groupby("INSURANCE_TYPE")["CLAIM_AMOUNT"].mean()
avg_claim


# %%
def claim_deviations(row):
    if current_date - row["TXN_DATE_TIME"] < pd.to_timedelta("30d"):
        return -1
    return 1 if avg_claim[row["INSURANCE_TYPE"]] / row["CLAIM_AMOUNT"] < 0.5 else 0


dfid_claim_deviation["CLAIM_DEVIATION"] = dfid_claim_deviation.apply(claim_deviations, axis=1)
dfid_claim_deviation["CLAIM_DEVIATION"].value_counts()


# %%


# %%
import numpy as np

df = pd.DataFrame(
    {
        "dates": pd.date_range(start="01-10-2022", periods=31, freq="d"),
        "count": range(31),
        "group": np.random.randint(0, 3, 31),
    }
)
# df.set_index('dates',inplace=True)
df.head()


# %%
df.rolling(window="3d", min_periods=3, closed="left", on="dates").sum().head()


# %%
df[df["group"] == 0].rolling("5d", 1, closed="left", on="dates").sum()


# %%
df.groupby("group").rolling(window="3d", closed="left", on="dates").mean()


# %%
df_claim_deviation_2 = df_insurance[["TXN_DATE_TIME", "INSURANCE_TYPE", "CLAIM_AMOUNT"]].copy()
df_claim_deviation_2["TXN_DATE_TIME"] = pd.to_datetime(df_claim_deviation_2["TXN_DATE_TIME"])
df_claim_deviation_2.describe(datetime_is_numeric=True)


# %%
df_claim_deviation_2


# %%
df_claim_deviation_2.set_index("TXN_DATE_TIME").groupby("INSURANCE_TYPE")["CLAIM_AMOUNT"].rolling(
    "30D", closed="left"
).mean().groupby(["INSURANCE_TYPE", "TXN_DATE_TIME"]).last()



