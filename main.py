import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import datetime
import yaml


def calc_future_value(P: float, r: float, n: int, t: int, a: int, M=0) -> pd.DataFrame:
    """Calculates the future value with compound interest for the following variables:
    P: Principal Amount
    r: expected annual interest rate as a fraction (e.g. .06 for 6%)
    t: time in years
    n: number of times interest is applied per year (e.g. 12 for monthly)
    a: starting age
    M: additional monthly contribution/deduction (optional)
    """
    data = []
    for period in range(t):
        amount = P * (np.power((1 + r / n), n * (period + 1))) + M * (
            np.power((1 + r / n), n * period + 1) - 1
        ) / (r / n)
        data.append({"Rate of Return": r, "Age": a + period, "Amount": amount})
    return pd.DataFrame(data=data, columns=["Age", "Rate of Return", "Amount"])


# Load configuration file
with open("sample_input.yaml") as config_file:
    config = yaml.safe_load(config_file)
print(config)
pd.options.display.float_format = "{:,.2f}".format
P = config.get("starting_principal")
M = config.get("monthly_contributions")
n = config.get("compound_frequency")
r = config.get("rate_of_return")

year = datetime.now().year
age = config.get("current_age")
retirement_age = config.get("retirement_age")
life_expectancy = config.get("life_expectancy")
t1 = retirement_age - age
t2 = life_expectancy - retirement_age
m_i_r = config.get("yearly_income_in_retirement")  # Pension, Social Security, etc...
m_s_r = config.get("yearly_spend_in_retirement")
I = float(m_i_r / 12)
E = float(m_s_r / 12)

# Pre-retirement calculations
df_pre = calc_future_value(P, r, n, t1, age, M)
P2 = df_pre.max().get("Amount")

# Post-retirement calculations
df_post = calc_future_value(P2, r, n, t2, age + t1, I - E)

results = pd.concat([df_pre, df_post])

print(results)

# Setup graph visuals and styling
sns.set_theme(rc={"figure.figsize": (14, 9)}, style="darkgrid")
plt.ticklabel_format(style="plain", axis="y")
ax = sns.lineplot(
    data=results, x="Age", y="Amount", hue="Rate of Return", palette="flare"
)

ax.set(
    title=f"Compound Interest Plot (Calculated Monthly, \${P:,d} Principal, \${M:,d} Monthly Contribution)",
    ylabel="Amount $",
)

ax.get_yaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

ax.get_xaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x)))
)

plt.show()
